"""Frozen, first-visit quotient values from committed training history only.

Equations follow v8.3.1 sections 4.1--4.3 literally. In particular the negative
conflict prior is a signed shaping potential, not a valid Beta probability;
the ordered lower floors have no upper clip and can produce values above one.
Neither same-bucket ordering nor its .02 navigation margin guarantees a
positive discounted TD across all observations or budget boundaries.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from .quotient import CELLS, _search
from .tracker import STAGES

FAMILIES = tuple(STAGES)
FAMILY_INDEX = {family: i for i, family in enumerate(FAMILIES)}
SHAPE = (6, 5, 2, 3)
ETA = (1.0, .85, .65)
ARRAY_CELLS = tuple((FAMILY_INDEX[f], p + 1, nav, b) for f, p, nav, b in CELLS)


def _readonly(array):
    """Bytes-backed arrays cannot be made writable by toggling numpy flags."""
    array = np.asarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


_mask = np.zeros(SHAPE, dtype=bool)
for _cell in ARRAY_CELLS:
    _mask[_cell] = True
REACHABLE_MASK = _readonly(_mask)
assert int(REACHABLE_MASK.sum()) == 96
del _mask, _cell


def get_step_bucket(step_1idx):
    if isinstance(step_1idx, bool) or not isinstance(step_1idx, (int, np.integer)) or not 1 <= step_1idx <= 50:
        raise ValueError('Decision step must be an integer in [1,50]')
    return 0 if step_1idx <= 10 else 1 if step_1idx <= 25 else 2


def canonicalize_quotient_cell(f_idx, p_t, b_nav_t, step_1idx):
    """Canonical dense-array index; non-search navigation collapses to one."""
    if isinstance(f_idx, str):
        if f_idx not in FAMILY_INDEX:
            raise ValueError('Unknown ALFWorld family')
        f_idx = FAMILY_INDEX[f_idx]
    if isinstance(f_idx, bool) or not isinstance(f_idx, (int, np.integer)) or not 0 <= f_idx < 6:
        raise ValueError('Invalid family index')
    family = FAMILIES[f_idx]
    if isinstance(p_t, bool) or not isinstance(p_t, (int, np.integer)) or not -1 <= p_t < STAGES[family]:
        raise ValueError('Quotient lookup requires a non-terminal progress state')
    if b_nav_t not in (0, 1):
        raise ValueError('Navigation indicator must be binary')
    return int(f_idx), int(p_t) + 1, int(b_nav_t) if _search(family, p_t) else 1, get_step_bucket(step_1idx)


def structured_prior():
    prior = np.zeros(SHAPE, dtype=np.float64)
    for (family, p, nav, b), idx in zip(CELLS, ARRAY_CELLS):
        prior[idx] = -.15 if p == -1 else min(.95, max(0., ETA[b] * (
            p / STAGES[family] + (.05 * (2 * nav - 1) if _search(family, p) else 0.))))
    return _readonly(prior)


PRIOR = structured_prior()


def project_values(raw):
    """Apply the specified ordered floors, with no extra clipping/projection."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != SHAPE or not np.isfinite(raw).all() or np.any(raw[~REACHABLE_MASK] != 0):
        raise ValueError('Malformed raw quotient array')
    result = raw.copy()
    for f_idx, family in enumerate(FAMILIES):
        for b in range(3):
            for p in range(STAGES[family]):
                if _search(family, p):
                    result[f_idx, p + 1, 1, b] = max(raw[f_idx, p + 1, 1, b], result[f_idx, p + 1, 0, b] + .02)
            for p in range(1, STAGES[family]):
                prev = result[f_idx, p, 1, b]
                if _search(family, p):
                    result[f_idx, p + 1, 0, b] = max(raw[f_idx, p + 1, 0, b], prev + .02)
                    result[f_idx, p + 1, 1, b] = max(raw[f_idx, p + 1, 1, b], result[f_idx, p + 1, 0, b] + .02, prev + .05)
                else:
                    result[f_idx, p + 1, 1, b] = max(raw[f_idx, p + 1, 1, b], prev + .05)
            result[f_idx, 0, 1, b] = min(raw[f_idx, 0, 1, b], result[f_idx, 1, 0, b] - .10, -.05)
    return _readonly(result)


def _validate_counts(counts, successes):
    n, s = np.asarray(counts), np.asarray(successes)
    if n.shape != SHAPE or s.shape != SHAPE or n.dtype.kind not in 'iu' or s.dtype.kind not in 'iu':
        raise ValueError('Counts must be integer arrays of quotient shape')
    if np.any(n < 0) or np.any(s < 0) or np.any(s > n) or np.any(n[~REACHABLE_MASK]) or np.any(s[~REACHABLE_MASK]):
        raise ValueError('Invalid first-visit counts or unreachable cell support')
    return np.asarray(n, dtype=np.int64), np.asarray(s, dtype=np.int64)


def _trajectory_cells(trace):
    if len(trace.steps) != len(trace.transitions) or not 1 <= len(trace.steps) <= 50:
        raise ValueError('Malformed trajectory length')
    cells = set()
    for t, (step, transition) in enumerate(zip(trace.steps, trace.transitions)):
        if step['step_id'] != t:
            raise ValueError('Expected consecutive original decision indices')
        before = transition.before
        if before.p == trace.task.K:
            continue
        cells.add(canonicalize_quotient_cell(trace.task.family, before.p, before.nav_bit, t + 1))
    return cells


def _donors(traces, af, pool_mode, *, iteration=None):
    if pool_mode not in {'mixed_only', 'all_groups'}:
        raise ValueError('Unknown donor pool mode')
    groups, seen = defaultdict(list), set()
    for trace in traces:
        if trace.source.get('split') != 'train' or 'trajectory' in trace.source:
            raise ValueError('Only explicitly marked training rollouts may enter the value table')
        if iteration is not None and trace.source.get('iteration_id', iteration) != iteration:
            raise ValueError('Donor rollout belongs to another iteration')
        if not trace.complete or not isinstance(trace.success, bool):
            raise ValueError('Donors require a completed binary terminal outcome')
        if trace.trajectory_uid in seen:
            raise ValueError('Duplicate original trajectory in donor pool')
        seen.add(trace.trajectory_uid)
        groups[trace.group_uid].append(trace)
    if set(groups) != set(af):
        raise ValueError('AF flags and donor task groups disagree')
    for group, members in groups.items():
        if not isinstance(af[group], bool) or af[group] != (not any(t.success for t in members)):
            raise ValueError('AF classification contradicts original rollout outcomes')
        if len({t.task_uid for t in members}) != 1:
            raise ValueError('Task group combines different tasks')
        if pool_mode == 'all_groups' or not af[group]:
            for trace in members:
                yield trace, _trajectory_cells(trace)


@dataclass(frozen=True, slots=True)
class FrozenValueSnapshot:
    iteration_id: int
    history_iterations: tuple
    donor_trajectories: int
    alpha0: float
    pool_mode: str
    counts: np.ndarray
    successes: np.ndarray
    raw: np.ndarray
    values: np.ndarray

    @property
    def diagnostics(self):
        n, s, raw, v = (a[REACHABLE_MASK] for a in (self.counts, self.successes, self.raw, self.values))
        rows = [dict(family=f, p=p, nav=nav, budget=b, n_hist=int(self.counts[idx]),
                     s_hist=int(self.successes[idx]), prior=float(PRIOR[idx]),
                     raw=float(self.raw[idx]), value=float(self.values[idx]))
                for (f, p, nav, b), idx in zip(CELLS, ARRAY_CELLS)]
        return dict(iteration_id=self.iteration_id, history_iterations=list(self.history_iterations),
                    pool_mode=self.pool_mode, donor_trajectories=self.donor_trajectories,
                    reachable_cells=96, finite_regularized_cells=int(np.isfinite(v).sum()),
                    empirically_observed_cells=int(np.count_nonzero(n)), prior_only_cells=int(np.count_nonzero(n == 0)),
                    minimum_regularized_support=float((n + self.alpha0).min()),
                    first_visit_count=int(n.sum()), projection_changed_cells=int(np.count_nonzero(v != raw)),
                    projected_above_one_cells=int(np.count_nonzero(v > 1)),
                    value_min=float(v.min()), value_max=float(v.max()), cells=rows,
                    value_semantics='signed shaping potentials; not calibrated success probabilities')

    def value_for_state(self, task, state, step_1idx):
        if state.p == task.K:
            return 1.0
        return float(self.values[canonicalize_quotient_cell(task.family, state.p, state.nav_bit, step_1idx)])

    def values_for_trace(self, trace):
        if not trace.complete or not trace.transitions:
            raise ValueError('Tier 2 requires a completed original rollout')
        result = [self.value_for_state(trace.task, transition.before, t + 1)
                  for t, transition in enumerate(trace.transitions)]
        # The current decision at step 50 has no nonterminal step-51 lookup.
        final = trace.transitions[-1].after
        result.append(1.0 if final.p == trace.task.K else 0.0)
        return result

    def calibration(self, traces, af, pool_mode=None):
        count = donors = successes = 0
        sum_value = sum_error = square_error = 0.
        for trace, cells in _donors(traces, af, pool_mode or self.pool_mode, iteration=self.iteration_id):
            donors += 1
            for cell in cells:
                v, y = float(self.values[cell]), int(trace.success)
                count += 1; successes += y; sum_value += v
                sum_error += v - y; square_error += (v - y) ** 2
        return dict(measurement='current training donors scored by frozen previous-window table; first-visit cells',
                    score_semantics='unclipped potential-versus-binary-outcome residual, not probability calibration',
                    donor_trajectories=donors, first_visit_pairs=count,
                    mean_value=sum_value / count if count else None,
                    mean_outcome=successes / count if count else None,
                    mean_signed_error=sum_error / count if count else None,
                    mean_squared_error=square_error / count if count else None)


def _make_snapshot(iteration, history_iterations, donor_trajectories, counts, successes, alpha0, pool_mode):
    n, s = _validate_counts(counts, successes)
    raw = np.zeros(SHAPE, dtype=np.float64)
    raw[REACHABLE_MASK] = (s[REACHABLE_MASK] + alpha0 * PRIOR[REACHABLE_MASK]) / (n[REACHABLE_MASK] + alpha0)
    return FrozenValueSnapshot(iteration, tuple(history_iterations), donor_trajectories, alpha0, pool_mode,
                               _readonly(n), _readonly(s), _readonly(raw), project_values(raw))


class QuotientValueTable:
    def __init__(self, *, window=4, alpha0=5.0, pool_mode='mixed_only'):
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise ValueError('Window must be a positive iteration count')
        if not math.isfinite(alpha0) or alpha0 <= 0 or pool_mode not in {'mixed_only', 'all_groups'}:
            raise ValueError('Invalid value-table parameters')
        self.window, self.alpha0, self.pool_mode = window, float(alpha0), pool_mode
        self.last_committed_iteration = 0
        self._history = []
        self.current_snapshot = None

    def snapshot(self, iteration):
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration != self.last_committed_iteration + 1:
            raise ValueError('Snapshot must follow the last committed iteration without gaps')
        if self.current_snapshot is not None and self.current_snapshot.iteration_id == iteration:
            return self.current_snapshot
        history = [r for r in self._history if iteration - self.window <= r['iteration_id'] < iteration]
        n, s = np.zeros(SHAPE, dtype=np.int64), np.zeros(SHAPE, dtype=np.int64)
        for r in history:
            n += r['counts']; s += r['successes']
        self.current_snapshot = _make_snapshot(iteration, [r['iteration_id'] for r in history],
            sum(r['donor_trajectories'] for r in history), n, s, self.alpha0, self.pool_mode)
        return self.current_snapshot

    def values_for_trace(self, trace):
        if self.current_snapshot is None:
            raise RuntimeError('Freeze the iteration table before computing credit')
        return self.current_snapshot.values_for_trace(trace)

    def commit(self, iteration, traces, af):
        if self.current_snapshot is None or self.current_snapshot.iteration_id != iteration or iteration != self.last_committed_iteration + 1:
            raise RuntimeError('Commit requires the matching uncommitted frozen iteration')
        traces = list(traces)
        n, s = np.zeros(SHAPE, dtype=np.int64), np.zeros(SHAPE, dtype=np.int64)
        donor_count = 0
        for trace, cells in _donors(traces, af, self.pool_mode, iteration=iteration):
            donor_count += 1
            for cell in cells:
                n[cell] += 1; s[cell] += int(trace.success)
        calibration = self.current_snapshot.calibration(traces, af, self.pool_mode)
        record = dict(iteration_id=iteration, donor_trajectories=donor_count,
                      counts=_readonly(n), successes=_readonly(s))
        self._history = (self._history + [record])[-self.window:]
        self.last_committed_iteration = iteration
        return calibration

    def state_dict(self):
        def counts_record(record):
            return {**record, 'counts': record['counts'].tolist(), 'successes': record['successes'].tolist()}
        snap = self.current_snapshot
        frozen = None if snap is None else counts_record(dict(iteration_id=snap.iteration_id,
            history_iterations=list(snap.history_iterations), donor_trajectories=snap.donor_trajectories,
            counts=snap.counts, successes=snap.successes))
        return dict(version='8.3.1-quotient-v1', window=self.window, alpha0=self.alpha0,
            pool_mode=self.pool_mode, last_committed_iteration=self.last_committed_iteration,
            cells_sha256=hashlib.sha256(json.dumps(CELLS).encode()).hexdigest(),
            history=[counts_record(r) for r in self._history], frozen_snapshot=frozen)

    def load_state_dict(self, state):
        for key in ['version', 'window', 'alpha0', 'pool_mode', 'cells_sha256']:
            if state.get(key) != self.state_dict()[key]:
                raise ValueError('Value-table checkpoint contract mismatch: ' + key)
        last = state['last_committed_iteration']
        if isinstance(last, bool) or not isinstance(last, int) or last < 0:
            raise ValueError('Invalid committed iteration')
        history = []
        for record in state['history']:
            n, s = _validate_counts(record['counts'], record['successes'])
            donors = record['donor_trajectories']
            if isinstance(donors, bool) or not isinstance(donors, int) or donors < int(n.max()):
                raise ValueError('Invalid donor trajectory count')
            history.append(dict(iteration_id=record['iteration_id'], donor_trajectories=donors,
                                counts=_readonly(n), successes=_readonly(s)))
        expected = list(range(max(1, last - self.window + 1), last + 1))
        if [r['iteration_id'] for r in history] != expected:
            raise ValueError('Missing, duplicated, or out-of-order history iterations')
        frozen = state['frozen_snapshot']
        snap = None
        if frozen is not None:
            i = frozen['iteration_id']
            if i not in (last, last + 1) or i < 1:
                raise ValueError('Frozen snapshot and committed history disagree')
            if frozen['history_iterations'] != list(range(max(1, i - self.window), i)):
                raise ValueError('Snapshot window includes invalid iteration identities')
            snap = _make_snapshot(i, frozen['history_iterations'], frozen['donor_trajectories'],
                frozen['counts'], frozen['successes'], self.alpha0, self.pool_mode)
        elif last:
            raise ValueError('Committed history lacks its frozen snapshot')
        self._history, self.last_committed_iteration, self.current_snapshot = history, last, snap
