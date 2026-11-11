"""Numeric and lifecycle oracles for the v8.3.1 historical quotient table."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from repro_v83.tracker import Task, STAGES
from repro_v83.traces import TrackedTrace, track_trace
from repro_v83.value_table import (
    ARRAY_CELLS, CELLS, FAMILIES, FAMILY_INDEX, PRIOR, REACHABLE_MASK, SHAPE,
    QuotientValueTable, canonicalize_quotient_cell, get_step_bucket, project_values,
)


def synthetic(uid, *, success=False, family='simple', group='group', states=None, iteration=1):
    task = Task(family, 'mug', 'desk')
    if states is None:
        states = [(0, 0)] * 50
    points = [SimpleNamespace(p=p, nav_bit=nav, won=False) for p, nav in states]
    final = SimpleNamespace(p=task.K if success else points[-1].p,
                            nav_bit=1 if success else points[-1].nav_bit, won=success)
    steps = [dict(step_id=i, action_uid=f'{uid}:{i}') for i in range(len(states))]
    transitions = [SimpleNamespace(before=before, after=after)
                   for before, after in zip(points, points[1:] + [final])]
    return TrackedTrace(task, group, group, uid, success, True, steps, transitions,
                        dict(split='train', iteration_id=iteration))


class QuotientTableTests(unittest.TestCase):
    def test_UT7_exact_96_cell_dense_mask_and_canonicalization(self):
        self.assertEqual(REACHABLE_MASK.shape, (6, 5, 2, 3))
        self.assertEqual(int(REACHABLE_MASK.sum()), 96)
        self.assertEqual([int(REACHABLE_MASK[f].sum()) for f in range(6)], [12, 15, 15, 15, 18, 21])
        for idx, (family, p, nav, b) in zip(ARRAY_CELLS, CELLS):
            step = [1, 11, 26][b]
            self.assertEqual(canonicalize_quotient_cell(family, p, nav, step), idx)
            self.assertTrue(REACHABLE_MASK[idx])
        self.assertEqual(canonicalize_quotient_cell(0, 1, 0, 1), (0, 2, 1, 0))
        for step, expected in [(1, 0), (10, 0), (11, 1), (25, 1), (26, 2), (50, 2)]:
            self.assertEqual(get_step_bucket(step), expected)
        for args in [(0, 2, 1, 1), (0, 1, 1, 51), (0, -2, 1, 1), (6, 0, 0, 1), (0, 0, 2, 1)]:
            with self.assertRaises(ValueError):
                canonicalize_quotient_cell(*args)

    def test_UT7_cold_start_prior_support_and_recovery(self):
        snap = QuotientValueTable().snapshot(1)
        self.assertEqual(snap.diagnostics['finite_regularized_cells'], 96)
        self.assertEqual(snap.diagnostics['empirically_observed_cells'], 0)
        self.assertEqual(snap.diagnostics['minimum_regularized_support'], 5.)
        self.assertEqual(snap.diagnostics['history_iterations'], [])
        np.testing.assert_allclose(snap.raw, PRIOR, rtol=0, atol=1e-16)
        for f, family in enumerate(FAMILIES):
            for b in range(3):
                conflict = snap.values[f, 0, 1, b]
                self.assertLessEqual(conflict, -.05)
                for nav in (0, 1):
                    self.assertGreater(.98 * snap.values[f, 1, nav, b] - conflict, 0)
                for p in range(1, STAGES[family]):
                    self.assertGreater(snap.values[f, p + 1, 1, b], snap.values[f, p, 1, b])

    def test_UT7_projection_numeric_oracle_with_empirical_inversion(self):
        raw = np.zeros(SHAPE)
        # Deliberately inverted stage ordering, including the second search.
        raw[5, 1, 0, 0] = .80
        raw[5, 1, 1, 0] = .10
        raw[5, 2, 1, 0] = .20
        raw[5, 3, 0, 0] = .30
        raw[5, 3, 1, 0] = .40
        raw[5, 4, 1, 0] = .50
        raw[5, 0, 1, 0] = .90
        v = project_values(raw)
        for idx, expected in [((5, 1, 0, 0), .80), ((5, 1, 1, 0), .82),
                              ((5, 2, 1, 0), .87), ((5, 3, 0, 0), .89),
                              ((5, 3, 1, 0), .92), ((5, 4, 1, 0), .97),
                              ((5, 0, 1, 0), -.05)]:
            self.assertAlmostEqual(v[idx], expected)

    def test_UT7_first_visit_counts_once_per_trajectory_cell(self):
        table = QuotientValueTable()
        frozen = table.snapshot(1)
        # Twenty-five consecutive visits to late-bucket cell count once.
        traces = [synthetic('win', success=True), synthetic('lose')]
        calibration = table.commit(1, traces, {'group': False})
        self.assertEqual(calibration['first_visit_pairs'], 6)
        self.assertEqual(int(frozen.counts.sum()), 0)
        self.assertEqual(frozen.values[0, 1, 0, 2], 0)
        snap = table.snapshot(2)
        self.assertEqual(snap.counts[0, 1, 0, 2], 2)
        self.assertEqual(snap.successes[0, 1, 0, 2], 1)
        self.assertAlmostEqual(snap.raw[0, 1, 0, 2], 1 / 7)
        self.assertEqual(int(snap.counts.sum()), 6)
        self.assertEqual(snap.history_iterations, (1,))
        # Freeze is immutable even if callers try re-enabling numpy writes.
        with self.assertRaises(ValueError):
            snap.values.setflags(write=True)

    def test_current_iteration_cannot_change_frozen_credit(self):
        table = QuotientValueTable()
        snap = table.snapshot(1)
        original = snap.values.tobytes()
        table.commit(1, [synthetic('win', success=True)], {'group': False})
        self.assertEqual(snap.values.tobytes(), original)
        self.assertIs(table.current_snapshot, snap)
        with self.assertRaises(RuntimeError):
            table.commit(1, [], {})
        with self.assertRaises(ValueError):
            table.snapshot(3)
        self.assertFalse(np.array_equal(table.snapshot(2).raw, snap.raw))

    def test_mixed_only_excludes_af_but_all_groups_includes_them(self):
        traces = [synthetic('w', success=True), synthetic('f'), synthetic('af', group='failed')]
        for mode, expected in [('mixed_only', 2), ('all_groups', 3)]:
            table = QuotientValueTable(pool_mode=mode)
            table.snapshot(1)
            table.commit(1, traces, {'group': False, 'failed': True})
            snap = table.snapshot(2)
            self.assertEqual(snap.counts[0, 1, 0, 0], expected)
            self.assertEqual(snap.successes[0, 1, 0, 0], 1)
            self.assertEqual(snap.donor_trajectories, expected)

    def test_history_exact_four_iterations_and_empty_donor_iteration(self):
        table = QuotientValueTable()
        for i in range(1, 7):
            snap = table.snapshot(i)
            self.assertEqual(snap.history_iterations, tuple(range(max(1, i - 4), i)))
            traces = [synthetic(f'{i}', success=i != 3, iteration=i)]
            table.commit(i, traces, {'group': i == 3})
        snap = table.snapshot(7)
        self.assertEqual(snap.history_iterations, (3, 4, 5, 6))
        self.assertEqual(snap.counts[0, 1, 0, 0], 3)
        self.assertEqual(len(table.state_dict()['history']), 4)

    def test_resume_reproduces_next_snapshot_and_rejects_corruption(self):
        table = QuotientValueTable()
        for i in range(1, 7):
            table.snapshot(i)
            table.commit(i, [synthetic(f'w{i}', success=True, iteration=i), synthetic(f'f{i}', iteration=i)], {'group': False})
        state = json.loads(json.dumps(table.state_dict(), allow_nan=False))
        restored = QuotientValueTable()
        restored.load_state_dict(state)
        self.assertEqual(restored.last_committed_iteration, 6)
        self.assertEqual(restored.current_snapshot.diagnostics, table.current_snapshot.diagnostics)
        self.assertEqual(restored.snapshot(7).diagnostics, table.snapshot(7).diagnostics)
        for mutate in [lambda s: s.update(alpha0=6.),
                       lambda s: s['history'].pop(0),
                       lambda s: s['history'][0]['counts'][0][0][0].__setitem__(0, 1),
                       lambda s: s['history'][0]['successes'][0][1][0].__setitem__(0, 99)]:
            invalid = copy.deepcopy(state); mutate(invalid)
            with self.assertRaises(ValueError):
                QuotientValueTable().load_state_dict(invalid)

    def test_rejects_eval_partial_duplicate_and_misclassified_donors(self):
        valid = synthetic('w', success=True)
        mutations = [lambda t: t.source.update(split='valid_seen'),
                     lambda t: t.source.update(iteration_id=2),
                     lambda t: setattr(t, 'complete', False)]
        for change in mutations:
            trace = copy.deepcopy(valid); change(trace)
            table = QuotientValueTable(); table.snapshot(1)
            with self.assertRaises(ValueError):
                table.commit(1, [trace], {'group': False})
            self.assertEqual(table.last_committed_iteration, 0)
        table = QuotientValueTable(); table.snapshot(1)
        with self.assertRaises(ValueError):
            table.commit(1, [valid, valid], {'group': False})
        with self.assertRaises(ValueError):
            table.commit(1, [valid], {'group': True})

    def test_success_timeout_and_next_step_budget_boundaries(self):
        snap = QuotientValueTable().snapshot(1)
        trace = synthetic('f', states=[(1, 1)] * 50)
        values = snap.values_for_trace(trace)
        self.assertEqual(len(values), 51)
        self.assertEqual(values[9], .5)
        self.assertEqual(values[10], .5 * .85)
        self.assertEqual(values[24], .5 * .85)
        self.assertEqual(values[25], .5 * .65)
        self.assertEqual(values[-1], 0.)
        trace.success = True
        trace.transitions[-1].after.p = trace.task.K
        self.assertEqual(snap.values_for_trace(trace)[-1], 1.)

    def test_literal_floors_do_not_claim_universal_td_sign_guarantee(self):
        raw = np.array(PRIOR)
        raw[0, 1, 0, 0] = 500 / 505
        projected = project_values(raw)
        self.assertGreater(projected[0, 1, 1, 0], 1.)
        self.assertLess(.98 * projected[0, 1, 1, 0] - projected[0, 1, 0, 0], 0.)
        # Floors apply independently per bucket, so cross-bucket progress
        # need not keep the same order when empirical support differs.
        self.assertLess(.98 * projected[0, 2, 1, 1] - projected[0, 1, 1, 0], 0.)


class RealTraceColdTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parent / 'reports/v83_20261030/replayed_traces.jsonl'
        cls.traces = {(r['failure_id'], r['method']): track_trace(r)
                      for r in (json.loads(line) for line in path.read_text().splitlines())}
        cls.snap = QuotientValueTable().snapshot(1)

    def credit(self, fid, method, step):
        trace = self.traces[(fid, method)]
        tr = trace.transitions[step - 1]
        values = self.snap.values_for_trace(trace)
        if tr.no_op:
            return -.05
        bonus = .1 if trace.task.family == 'look_at_obj_in_light' and tr.first_lamp_activation and tr.before.p <= 1 else 0.
        return .98 * values[step] - values[step - 1] + bonus

    def test_real_conflict_recovery_lamp_and_terminal_cases(self):
        for fid, method, step, positive in [
            ('F03', 'RePro', 3, False), ('F22', 'GiGPO', 7, True),
            ('F27', 'GiGPO', 4, False), ('F27', 'GiGPO', 14, True),
            ('F12', 'RePro', 2, True), ('F01', 'RePro', 7, False),
            ('F12', 'GiGPO', 10, True), ('F31', 'GiGPO', 24, True),
            ('F31', 'RePro', 15, False),
        ]:
            with self.subTest(fid=fid, method=method, step=step):
                delta = self.credit(fid, method, step)
                self.assertGreater(delta, 0.) if positive else self.assertLess(delta, 0.)
        for (fid, method), trace in self.traces.items():
            if trace.success:
                self.assertGreater(self.credit(fid, method, len(trace.steps)), 0.)


if __name__ == '__main__':
    unittest.main()
