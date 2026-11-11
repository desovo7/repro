import copy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PACKAGE = Path(os.environ.get('POLARITY_PACKAGE', HERE / 'webshop'))
spec = importlib.util.spec_from_file_location('polarity_ws', PACKAGE / '__init__.py', submodule_search_locations=[str(PACKAGE)])
module = importlib.util.module_from_spec(spec)
sys.modules['polarity_ws'] = module
spec.loader.exec_module(module)
core = importlib.import_module('polarity_ws.core')
credit = importlib.import_module('polarity_ws.r2_credit')
fixtures = importlib.import_module('polarity_ws.test_webshop')
aligned_tests = importlib.import_module('polarity_ws.test_r2_aligned')
FULL = core.Config(gate_invalid_noop=True, return_credit_mode='neutral', neutral_terminal_buy=True)


class InterventionTests(unittest.TestCase):
    def test_format_gate_does_not_relabel_execution_or_mutate_states(self):
        raw = fixtures.fixture(long=True)
        raw['steps'][1]['format_valid'] = False
        before = copy.deepcopy(raw)
        baseline = core.parse_trace(raw, core.Config())
        candidate = core.parse_trace(raw, FULL)
        self.assertFalse(candidate['trace'].format_valids[1])
        self.assertFalse(candidate['noops'][1])
        for key in ['states', 'noops', 'coverage']:
            self.assertEqual(baseline[key], candidate[key])
        self.assertEqual(raw, before)
        del raw['steps'][1]['format_valid']
        with self.assertRaisesRegex(ValueError, 'format_valid'):
            core.parse_trace(raw, FULL)

    def test_return_gate_requires_actual_outward_transition(self):
        raw = aligned_tests.sequence([
            ('click[description]', fixtures.DETAIL, fixtures.CLICK),
            ('click[< prev]', fixtures.ITEM, ['< prev', 'back to search']),
            ('click[< prev]', fixtures.RESULTS, fixtures.CLICK),
            ('click[b000000001]', fixtures.ITEM, ['b000000001']),
            ('click[buy now]', fixtures.END, fixtures.CLICK),
        ])
        t = core.parse_trace(raw, FULL)['trace']
        self.assertFalse(t.return_navs[3])  # detail -> item is not candidate reselection
        self.assertTrue(t.return_navs[4])   # item -> results
        self.assertEqual(sum(t.return_navs), 1)
        self.assertTrue(t.terminal_buys[-1])

    def test_neutral_gate_cuts_propagation_and_stays_zero_after_permutation(self):
        raws = fixtures.batch()
        for r in raws:
            r['steps'][1]['format_valid'] = False
        cfg = core.Config(gate_invalid_noop=True, return_credit_mode='neutral', neutral_terminal_buy=True, control='permute')
        e = core.CreditEngine(cfg)
        result = e.prepare(raws, 1)
        self.assertTrue(result['gate_reasons'])
        for uid in result['gate_reasons']:
            self.assertEqual([result[k][uid] for k in ['raw', 'deltas', 'centered']], [0., 0., 0.])
        uids = list(result['centered'])
        injection, _ = core.token_injections(result, uids, [i % 7 + 1 for i in range(len(uids))], cfg)
        for uid, value in zip(uids, injection):
            if uid in result['gate_reasons']:
                self.assertEqual(value, 0.)

    def test_neutral_return_is_stronger_than_delta_clipping(self):
        traces = [core.parse_trace(r, FULL)['trace'] for r in fixtures.batch()]
        # Controlled credit-core example: a return is followed by later progress.
        for t in traces:
            t.return_navs[1] = True
        def potential(t, _tables, _cfg):
            return [0., .6, 0.] + [.2] * (len(t.steps) - 2), 0
        with patch.object(credit, 'potentials', potential):
            neutral = credit.credit(traces, [], core.Config(), FULL, 8, 15)
            clip_cfg = core.Config(return_credit_mode='negative_delta_clip')
            clipped = credit.credit(traces, [], core.Config(), clip_cfg, 8, 15)
        returns = [t.steps[1]['action_uid'] for t in traces]
        self.assertTrue(all(neutral['raw'][u] == neutral['centered'][u] == 0 for u in returns))
        self.assertTrue(any(clipped['raw'][u] > 0 for u in returns))

    def test_mixed_groups_and_checkpoint_config_contract(self):
        e = core.CreditEngine(FULL)
        result = e.prepare(fixtures.batch(wins=3), 1)
        self.assertTrue(all(v == 0 for v in result['centered'].values()))
        self.assertFalse(result['gate_reasons'])
        e.commit(1)
        state = json.loads(json.dumps(e.state_dict()))
        restored = core.CreditEngine(FULL)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            core.CreditEngine(core.Config()).load_state_dict(state)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    for mod in [fixtures, aligned_tests, sys.modules[__name__]]:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(mod))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
