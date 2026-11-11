"""Causal WebShop interface for the deployed ALFWorld R2 credit core.

Only instructions, nonterminal text, executed actions and current clickables
enter this interface. C3 covers the frozen first visible search slate only.
"""
from collections import Counter
import json
import math
import re
import unicodedata

from .r2_credit import Trace

NAV = {'back to search', '< prev', 'next >', 'description', 'features', 'reviews', 'attributes', 'buy now', 'search'}


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def raw_parts(text):
    # simple-mode text wraps each field in quotes; retain spaces within values.
    return [x[1:-1] if len(x) >= 2 and x[0] == x[-1] == "'" else x
            for x in str(text).split(' [SEP] ')]


def parts(text):
    return [norm(x) for x in raw_parts(text)]


def price_interval(text):
    number = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
    m = re.fullmatch(r'\$?(' + number + r')(?:\s+to\s+\$?(' + number + r'))?', norm(text))
    if m:
        values = [float(v.replace(',', '')) for v in m.groups() if v is not None]
        if all(math.isfinite(v) for v in values):
            return min(values), max(values)
    return None


def visible_product(observation):
    fields = parts(observation)
    for i, field in enumerate(fields):
        if field.startswith('price:'):
            return (fields[i-1] if i else ''), price_interval(field[6:].strip())
    return '', None


def price_relation(interval, limit):
    if interval is None or limit is None:
        return 'unknown'
    return 'within' if interval[1] <= limit else 'over' if interval[0] > limit else 'unknown'


def page_type(observation, previous='unknown', action=''):
    fields = parts(observation)
    if 'buy now' in fields:
        return 'item'
    if any(re.fullmatch(r'page \d+ \(total results: [\d,]+\)', f) for f in fields):
        return 'results'
    if 'search' in fields:
        return 'home'
    if action in {'click[description]', 'click[features]', 'click[reviews]', 'click[attributes]'}:
        return 'detail'
    return previous if previous == 'detail' else 'unknown'


def instruction_slots(instruction):
    """Conservative structured `with category: value, and category: value`."""
    text = norm(instruction)
    boundary = r'(?:\bwith\s+|,\s*(?:and\s+)?)([a-z][a-z0-9 _/-]{0,45}):\s*'
    matches = list(re.finditer(boundary, text))
    budget = re.search(r'\bprice lower than (\d+(?:\.\d+)?) dollars', text)
    slots, conflicts = {}, set()
    for i, match in enumerate(matches):
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        budget_follows = budget is not None and match.end() <= budget.start() < end
        if budget_follows:
            end = budget.start()
        value = re.sub(r',\s*(?:and\s*)?$', '', text[match.end():end]).strip()
        if budget_follows:
            value = re.sub(r'\s+and$', '', value).strip()
        category = norm(match[1])
        if not value or category in slots:
            conflicts.add(category)
        slots[category] = value
    for category in conflicts:
        slots[category] = None
    limit = float(budget[1]) if budget else None
    skeleton = json.dumps([sorted(slots), limit is not None], separators=(',', ':'))
    return slots, limit, skeleton, conflicts


def option_groups(observation, available):
    """Parse a nonclickable category followed by its visible clickable values."""
    fields = raw_parts(observation)
    clickable = {str(x).lower() for x in (available or {}).get('clickables', [])}
    groups = {}
    i = 0
    while i < len(fields):
        raw, category = fields[i], norm(fields[i])
        if raw.lower() in clickable or category in NAV or category.startswith(('price:', 'rating:')):
            i += 1
            continue
        j = i + 1
        values = []
        while j < len(fields) and fields[j].lower() in clickable and norm(fields[j]) not in NAV:
            values.append(fields[j].lower())
            j += 1
        if values:
            # Repeated/ambiguous headings are unavailable, never guessed.
            groups[category] = tuple(values) if category not in groups else ()
        i = j
    return groups


def first_slate(observation, available):
    fields = raw_parts(observation)
    clickable = {str(x).lower() for x in (available or {}).get('clickables', [])}
    candidates = [(i, field.lower()) for i, field in enumerate(fields)
                  if field.lower() in clickable and field.lower() not in NAV
                  and re.fullmatch(r'[a-z0-9]{10}', field.lower())]
    pool = {}
    for j, (i, product) in enumerate(candidates):
        end = candidates[j+1][0] if j+1 < len(candidates) else len(fields)
        # Some native results omit the title. Stay inside this candidate's
        # visible block so a missing price cannot borrow the next item's price.
        prices = [price_interval(norm(f)[6:].strip() if norm(f).startswith('price:') else f)
                  for f in fields[i+1:end] if norm(f).startswith(('$', 'price:'))]
        interval = prices[0] if len(prices) == 1 else None
        # Candidate class is fixed BEFORE opening and independent of the goal.
        cls = 'unknown' if interval is None else ('single' if interval[0] == interval[1] else 'range')
        pool[product] = (cls if product not in pool else 'unknown') + ' ' + product
    return pool


def parse_trace(raw, config):
    steps = raw['steps']
    if not 1 <= len(steps) <= config.horizon or any(s['step_id'] != i for i, s in enumerate(steps)):
        raise ValueError('Invalid original decision sequence')
    if any(s['environment_done'] for s in steps[:-1]):
        raise ValueError('Postterminal decision')
    if bool(steps[-1]['won']) != bool(raw['success']):
        raise ValueError('Success disagrees with final active decision')
    complete = bool(steps[-1]['environment_done'] or len(steps) == config.horizon)
    if not complete:
        raise ValueError('Early truncation cannot be called all-failed')
    requested, limit, skeleton, conflicts = instruction_slots(raw['instruction'])
    t = Trace(str(raw['task_uid']), str(raw['group_uid']), str(raw['trajectory_uid']),
              bool(raw['success']), complete, steps, raw['instruction'], (),
              skeleton=skeleton, slots={role: i for i, role in enumerate(sorted(requested))})
    page = page_type(steps[0]['observation_before'])
    if page != 'home':
        raise ValueError('Rollout must start at the reset search page')
    selected, groups, pool, opened = {}, {}, None, set()
    interval, product = None, None
    audit = Counter()
    audit['conflicting_instruction_roles'] = len(conflicts)
    audit['unparsed_instruction_roles'] = int(not requested)
    audit['requested_roles'] = len(requested)

    def key():
        statuses = []
        for role in sorted(requested):
            goal = requested[role]
            values = groups.get(role)
            if goal is None or values is None or not values:
                statuses.append('unresolved')
                reason = 'conflicting_request' if goal is None else 'category_unavailable' if values is None else 'ambiguous_heading'
                audit['unknown_slot_' + reason] += 1
            elif role in selected:
                statuses.append('unresolved' if selected[role] is None else
                                'selected_requested' if norm(selected[role]) == goal else 'selected_other')
                audit['unknown_slot_ambiguous_selection'] += int(selected[role] is None)
            elif goal in {norm(v) for v in values}:
                statuses.append('available_unselected')
            else:
                statuses.append('unresolved')
                audit['unknown_slot_requested_value_unavailable'] += 1
        relation = price_relation(interval, limit)
        if relation == 'unknown' and page in {'item', 'detail'}:
            reason = 'no_budget' if limit is None else 'unparsed_interval' if interval is None else 'straddling_interval'
            audit['unknown_price_' + reason] += 1
        return json.dumps([page, relation, statuses], separators=(',', ':'))

    t.states, t.reveals = [key()], [[]]
    previous = steps[0]['observation_before']
    for i, step in enumerate(steps):
        if step['observation_before'] != previous:
            raise ValueError('Discontinuous observation chain')
        available = step['available_before']
        clickables = {str(x).lower() for x in available['clickables']}
        action = str(step['action']).lower()
        parsed = re.match(r'(.+)\[(.+)\]', action)  # upstream grammar, no whitespace repair
        verb, arg = parsed.groups() if parsed else ('', '')
        executable = verb == 'search' or (verb == 'click' and arg in clickables and arg != 'search')
        noop = not executable
        new = []
        # Never parse terminal HTML, even if the buy failed.
        if not step['environment_done']:
            after = step['observation_after']
            after_page = page_type(after, page, f'{verb}[{arg}]')
            after_available = step.get('available_after')
            if after_available is None and i+1 < len(steps) and steps[i+1]['observation_before'] == after:
                after_available = steps[i+1]['available_before']
            if after_available is None and after == step['observation_before']:
                after_available = available
            if after_available is None:
                audit['missing_current_clickables'] += 1
            if page == 'item' and executable and verb == 'click' and arg not in NAV:
                roles = [role for role, values in groups.items() if arg in values]
                if len(roles) == 1:
                    role = roles[0]
                    noop = selected.get(role) == arg
                    selected[role] = arg
                    audit['repeated_accepted_option'] += int(noop)
                else:
                    audit['ambiguous_accepted_option'] += 1
                    # An accepted ambiguous click can overwrite any matching role.
                    for role in roles:
                        selected[role] = None
            opening = page == 'results' and after_page == 'item' and executable and verb == 'click'
            if after_page in {'home', 'results'} or opening:
                selected, groups, interval, product = {}, {}, None, None
            if after_page == 'results' and pool is None:
                pool = first_slate(after, after_available)
            if after_page == 'item':
                _, interval = visible_product(after)
                groups = option_groups(after, after_available)
                audit['unknown_price_item_states'] += int(price_relation(interval, limit) == 'unknown')
            if opening:
                product = arg
                if pool is not None and product in pool and product not in opened:
                    entity = pool[product]
                    opened.add(product)
                    new = [entity]
                    seen = tuple(role for role, goal in requested.items() if goal is not None
                                 and goal in {norm(v) for v in groups.get(role, ())})
                    t.sightings.append((entity, seen))
                elif pool is None or product not in pool:
                    audit['opens_outside_first_slate'] += 1
            page = after_page
            t.states.append(key())
        else:
            t.states.append(t.states[-1])
        t.noops.append(noop)
        t.reveals.append(new)
        previous = step['observation_after']
    t.room = tuple((pool or {}).values())
    audit['candidate_pool_size'] = len(t.room)
    audit['first_reveals'] = len(t.sightings)
    audit['positive_sightings'] = sum(len(v) for _, v in t.sightings)
    count, coverage = 0, []
    for new in t.reveals:
        count += len(new)
        coverage.append(count / len(t.room) if t.room else 0.0)
    return dict(raw=raw, trace=t, states=t.states, noops=t.noops, coverage=coverage, audit=dict(audit))
