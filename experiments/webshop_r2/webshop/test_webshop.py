"""Behavioral checks for the candidate; no environment/GPU is started."""
import copy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .core import Config, CreditEngine, parse_trace, shuffle_ranks, token_injections
from .integration import Runtime, captured_traces

HOME = "'Search'"
RESULTS = "'Back to Search' [SEP] 'Page 1 (Total results: 20)' [SEP] 'Next >' [SEP] 'b000000001' [SEP] 'Red cotton shirt' [SEP] '$20'"
ITEM = "'Back to Search' [SEP] '< Prev' [SEP] 'color' [SEP] 'red' [SEP] 'blue' [SEP] 'Red cotton shirt' [SEP] 'Price: $20' [SEP] 'Rating: N.A.' [SEP] 'Description' [SEP] 'Features' [SEP] 'Reviews' [SEP] 'Buy Now'"
DETAIL = "'Back to Search' [SEP] '< Prev' [SEP] 'Red cotton shirt is washable'"
END = 'Thank you! Your score 0.25 Target asin secret options red'
CLICK = ['back to search', '< prev', 'red', 'blue', 'description', 'features', 'reviews', 'buy now']


def fixture(uid='0', success=False, long=False, iteration=1):
    transitions = [(HOME, 'search[red shirt]', RESULTS, []),
                   (RESULTS, 'click[b000000001]', ITEM, ['b000000001', 'next >', 'back to search'])]
    if long:
        transitions += [(ITEM, 'click[red]', ITEM, CLICK), (ITEM, 'click[red]', ITEM, CLICK)]
    transitions += [(ITEM, 'click[buy now]', END, CLICK)]
    steps = [dict(step_id=i, action_uid=f'{iteration}:{uid}:{i}', observation_before=before,
                  observation_after=after, action=action, available_before=dict(has_search_bar=before==HOME, clickables=clicks),
                  won=success and i==len(transitions)-1, environment_done=i==len(transitions)-1,
                  format_valid=True, official_environment_reward=10.0 if success and i==len(transitions)-1 else 0.0)
             for i, (before, action, after, clicks) in enumerate(transitions)]
    return dict(run_id='test', iteration_id=iteration, split='train', task_uid='task', group_uid='group',
                trajectory_uid=uid, instruction='I want a red cotton shirt, and price lower than 30.00 dollars',
                success=success, steps=steps)


def batch(iteration=1, wins=0):
    return [fixture(str(i), i < wins, bool(i % 2), iteration) for i in range(8)]


class CoreTests(unittest.TestCase):
    def test_terminal_target_and_partial_score_do_not_change_features_or_credit(self):
        a = batch()
        b = copy.deepcopy(a)
        for trace in b:
            trace['steps'][-1]['observation_after'] = 'goal target blue price 0 SUCCESS score 1.0'
            trace['steps'][-1]['task_score'] = 1.0
        ea, eb = CreditEngine(), CreditEngine()
        self.assertEqual(ea.prepare(a, 1), eb.prepare(b, 1))

    def test_option_click_with_identical_text_is_not_noop(self):
        raw = fixture(long=True)
        t = parse_trace(raw, Config())
        self.assertFalse(t['noops'][2])
        self.assertNotEqual(t['states'][2], t['states'][3])
        self.assertEqual(t['states'][3], t['states'][4])
        self.assertEqual(t['coverage'][2], t['coverage'][4])

    def test_format_invalid_executable_action_is_not_declared_noop(self):
        raw = fixture(long=True)
        raw['steps'][2]['format_valid'] = False
        self.assertFalse(parse_trace(raw, Config())['noops'][2])

    def test_invalid_option_argument_is_noop(self):
        raw = fixture(long=True)
        raw['steps'][2]['action'] = 'click[ red ]'
        t = parse_trace(raw, Config())
        self.assertTrue(t['noops'][2])
        self.assertEqual(t['states'][2], t['states'][3])

    def test_item_features_grounded_in_instruction(self):
        a = fixture()
        b = copy.deepcopy(a)
        b['instruction'] = 'blue silk socks, and price lower than 10.00 dollars'
        ka = json.loads(parse_trace(a, Config())['states'][2])
        kb = json.loads(parse_trace(b, Config())['states'][2])
        self.assertGreater(ka[1], kb[1])
        self.assertEqual((ka[2], kb[2]), ('within', 'over'))

    def test_horizon_is_complete_without_terminal_state(self):
        raw = fixture()
        raw['steps'] = raw['steps'][:2]
        for i in range(2, 15):
            raw['steps'].append(dict(raw['steps'][1], step_id=i, action_uid=f'x:{i}', observation_before=ITEM,
                                    observation_after=ITEM, action='click[invalid]', available_before=dict(clickables=CLICK)))
        t = parse_trace(raw, Config())
        self.assertEqual(len(t['states']), 16)
        raw['steps'].pop()
        with self.assertRaises(ValueError):
            parse_trace(raw, Config())

    def test_price_equality_matches_native_boundary(self):
        raw = fixture()
        raw['instruction'] = 'shirt, and price lower than 20.00 dollars'
        state = json.loads(parse_trace(raw, Config())['states'][2])
        self.assertEqual(state[2], 'within')

    def test_rejects_discontinuous_or_postterminal_steps(self):
        raw = fixture()
        raw['steps'][1]['observation_before'] = 'different'
        with self.assertRaises(ValueError):
            parse_trace(raw, Config())
        raw = fixture()
        raw['steps'][0]['environment_done'] = True
        with self.assertRaises(ValueError):
            parse_trace(raw, Config())

    def test_mixed_group_receives_exactly_zero(self):
        engine = CreditEngine()
        result = engine.prepare(batch(wins=3), 1)
        self.assertFalse(result['af']['group'])
        self.assertTrue(all(v == 0 for v in result['centered'].values()))
        self.assertEqual(result['diagnostics']['donor_successes'], 0)
        engine.commit(1)
        result = engine.prepare(batch(iteration=2), 2)
        self.assertEqual(result['diagnostics']['donor_successes'], 3)
        self.assertGreater(result['diagnostics']['ranked_states'], 0)

    def test_current_batch_cannot_self_label(self):
        engine = CreditEngine()
        engine.prepare(batch(wins=8), 1)
        self.assertEqual(engine.ranks(), ({}, 0, 0))
        with self.assertRaises(ValueError):
            engine.state_dict()
        engine.commit(1)
        self.assertEqual(engine.ranks()[2], 8)

    def test_rejects_validation_and_incomplete_groups(self):
        raws = batch()
        raws[0]['split'] = 'validation'
        with self.assertRaises(ValueError):
            CreditEngine().prepare(raws, 1)
        with self.assertRaises(ValueError):
            CreditEngine().prepare(batch()[:-1], 1)

    def test_rejects_duplicate_ids(self):
        raws = batch()
        raws[1]['steps'][0]['action_uid'] = raws[0]['steps'][0]['action_uid']
        with self.assertRaises(ValueError):
            CreditEngine().prepare(raws, 1)

    def test_resume_exact_next_credits_and_bounded_history(self):
        cfg = Config(window=2)
        engine = CreditEngine(cfg)
        for i in range(1, 5):
            engine.prepare(batch(iteration=i, wins=3), i)
            engine.commit(i)
        state = json.loads(json.dumps(engine.state_dict()))
        self.assertEqual(len(state['history']), 2)
        loaded = CreditEngine(cfg)
        loaded.load_state_dict(state)
        self.assertEqual(engine.prepare(batch(iteration=5), 5), loaded.prepare(batch(iteration=5), 5))
        with self.assertRaises(ValueError):
            CreditEngine(Config(window=3)).load_state_dict(state)

    def test_permutation_preserves_total_centered_marginals(self):
        real = CreditEngine()
        placebo = CreditEngine(Config(control='permute'))
        for engine in (real, placebo):
            engine.prepare(batch(wins=8), 1)
            engine.commit(1)
        a = real.prepare(batch(iteration=2), 2)
        b = placebo.prepare(batch(iteration=2), 2)
        self.assertEqual(sorted(a['centered'].values()), sorted(b['centered'].values()))

    def test_baseline_disabled_and_mask_weighted_copying(self):
        engine = CreditEngine(Config(enabled=False))
        result = engine.prepare(batch(), 1)
        self.assertTrue(all(v == 0 for v in result['centered'].values()))
        result = dict(centered={'a': -1., 'b': 1., 'c': 0.}, af={'af': True, 'mixed': False},
                      action_groups={'a': 'af', 'b': 'af', 'c': 'mixed'})
        uids, weights = ['b', 'c', 'a', 'b'], [3, 10, 1, 3]
        values, metrics = token_injections(result, uids, weights, Config())
        self.assertEqual(values[0], values[3])
        self.assertEqual(values[1], 0)
        self.assertLess(values[2], 0)
        self.assertAlmostEqual(metrics['token_std_before_scale'], (24/49)**0.5)

    def test_ties_share_rank(self):
        e = CreditEngine(Config(min_success_support=1))
        e.history.append(dict(successes=1, order={'a': [0.5, 1], 'b': [0.5, 1]}))
        ranks, levels, _ = e.ranks()
        self.assertEqual(ranks, {'a': 1, 'b': 1})
        self.assertEqual(levels, 1)

    def test_rank_shuffle_preserves_support_and_multiset_and_replays(self):
        ranks = {'a':1, 'b':1, 'c':2, 'd':3}
        shuffled, changed = shuffle_ranks(ranks, 7, 8)
        self.assertEqual(set(shuffled), set(ranks))
        self.assertEqual(sorted(shuffled.values()), sorted(ranks.values()))
        self.assertGreater(changed, 0)
        self.assertEqual(shuffle_ranks(dict(reversed(list(ranks.items()))), 7, 8), (shuffled, changed))
        self.assertEqual(shuffle_ranks({'a':1, 'b':1}, 7, 8), ({'a':1, 'b':1}, 0))

    def test_rank_control_resume_and_coverage_unchanged(self):
        cfg = Config(control='rank_shuffle')
        control = CreditEngine(cfg)
        original = CreditEngine()
        for engine in (control, original):
            engine.prepare(batch(wins=8), 1)
            engine.commit(1)
        resumed = CreditEngine(cfg)
        resumed.load_state_dict(json.loads(json.dumps(control.state_dict())))
        a = control.prepare(batch(iteration=2), 2)
        b = original.prepare(batch(iteration=2), 2)
        self.assertEqual(a, resumed.prepare(batch(iteration=2), 2))
        self.assertEqual(control.ranks(), original.ranks())
        self.assertEqual(a['diagnostics']['rank_coverage_fraction'], b['diagnostics']['rank_coverage_fraction'])
        self.assertGreater(a['diagnostics']['rank_labels_changed'], 0)
        self.assertNotEqual(a['raw'], b['raw'])
        self.assertEqual(parse_trace(fixture(), cfg)['coverage'], parse_trace(fixture(), Config())['coverage'])

    def test_dose_metrics_use_actual_af_token_weights(self):
        result = dict(centered={'a':-1., 'b':2., 'c':0.}, af={'af':True, 'other':False},
                      action_groups={'a':'af', 'b':'af', 'c':'other'})
        delta, metrics = token_injections(result, ['a','b','b','c'], [1,3,3,100], Config(max_abs_injection=.1))
        self.assertEqual(delta, [-.1,.1,.1,0.])
        self.assertEqual(metrics['af_response_tokens'], 7)
        self.assertEqual(metrics['af_nonzero_tokens'], 7)
        self.assertEqual(metrics['af_clipped_token_fraction'], 1)
        self.assertAlmostEqual(metrics['af_weighted_rms'], .1)
        self.assertAlmostEqual(metrics['af_weighted_mean'], .5/7)

    def test_collector_uses_last_active_outcome_not_auto_reset_outcome(self):
        raw = fixture()
        rows, records = [], []
        for s in raw['steps']:
            rows.append(dict(active_masks=True, uid='g', traj_uid='t', anchor_obs=s['observation_before'], rewards=0.0))
            records.append([s])
        rows.append(dict(active_masks=False, uid='g', traj_uid='t', anchor_obs='new task', rewards=10.0))
        records.append([dict(raw['steps'][-1], won=True)])
        capture = SimpleNamespace(task_ids=['task'], instructions=[raw['instruction']], initial=[HOME], steps=records)
        traces = captured_traces([rows], capture, 'test', 1, True)
        self.assertFalse(traces[0]['success'])
        self.assertEqual(len(traces[0]['steps']), 3)
        self.assertTrue(all(row['anchor_obs'] == s['observation_before'] for row, s in zip(rows, raw['steps'])))

    def test_validation_never_changes_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(native_persistence=SimpleNamespace(identity=SimpleNamespace(run_id='test')),
                                  trainer=SimpleNamespace(default_local_dir=tmp+'/checkpoints'), gradus_webshop=asdict(Config()))
            runtime = Runtime(cfg)
            before = runtime.engine.state_dict()
            runtime.record_validation(batch(wins=8), 0)
            self.assertEqual(before, runtime.engine.state_dict())


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch optional locally; checked on A800 CPU')
class TensorTests(unittest.TestCase):
    def test_actual_tensor_mask_copy_and_native_immutability(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            cfg = SimpleNamespace(native_persistence=SimpleNamespace(identity=SimpleNamespace(run_id='test')),
                                  trainer=SimpleNamespace(default_local_dir=tmp+'/checkpoints'), gradus_webshop=asdict(Config()))
            runtime = Runtime(cfg)
            runtime.prepare(batch(wins=8), 1)
            runtime.engine.commit(1)
            runtime.phase = 'committed'
            runtime.prepare(batch(iteration=2), 2)
            ids = list(runtime.result['centered'])
            ids = ids[::-1] + ids[:2]
            base = torch.full((len(ids), 4), 0.25)
            rewards = torch.arange(len(ids), dtype=torch.float32)
            mask = torch.tensor([[1,1,0,0]]*len(ids))
            b = SimpleNamespace(batch={'advantages':base.clone(), 'response_mask':mask, 'rewards':rewards.clone()},
                                non_tensor_batch={'ws_action_uid':ids}, meta_info={})
            runtime.inject(b)
            self.assertTrue(torch.equal(b.batch['advantages'][:, 2:], base[:, 2:]))
            self.assertTrue(torch.equal(b.batch['rewards'], rewards))
            with self.assertRaises(ValueError):
                runtime.inject(b)
            # A failed/incomplete actor audit cannot advance donor state.
            with self.assertRaises(ValueError):
                runtime.commit(SimpleNamespace(meta_info={'metrics':{}}))
            self.assertEqual(runtime.engine.last_iteration, 1)


class RetentionTests(unittest.TestCase):
    def test_orphan_validation_does_not_block_restart(self):
        from .native_retention import CheckpointRetention
        identity = dict(run_id='orphan-test', config_hash='c', model_contract_hash='m')
        with tempfile.TemporaryDirectory() as tmp:
            keeper = CheckpointRetention(tmp, identity)
            keeper.record_validation(1, 1.0)
            restored = CheckpointRetention(tmp, identity)
            self.assertEqual(restored.state['history'], [])
            self.assertEqual(restored.state['best'], [])
            self.assertIsNone(restored.state['latest'])
            self.assertEqual(restored.score_for(1), 1.0)

    def test_archive_published_before_index_recovers_after_crash(self):
        from unittest.mock import patch
        from . import native_retention as retention
        identity = dict(run_id='crash-test', config_hash='c', model_contract_hash='m')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keeper = retention.CheckpointRetention(root, identity)
            folder = root/'global_step_1'
            for name in retention.REQUIRED:
                path = folder/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'stand-in-for-policy-check')
            runtime = dict(identity, iteration_id=1, gradus_state={'last_iteration':1})
            retention.write_json(folder/'native_runtime.json', runtime)
            retention.write_json(folder/'NATIVE_COMPLETE.json', dict(runtime,
                runtime_sha256=retention.digest(folder/'native_runtime.json'),
                rng_sha256=retention.digest(folder/'native_driver_rng.pt')))
            original_write = retention.write_json
            def fail_index(path, value):
                if Path(path) == keeper.index:
                    raise OSError('simulated interruption before atomic index commit')
                return original_write(path, value)
            with patch.object(retention, 'write_json', side_effect=fail_index):
                with self.assertRaises(OSError):
                    keeper.preserve(folder, .8)
            restored = retention.CheckpointRetention(root, identity)
            self.assertEqual(restored.state['best'], [{'step':1,'score':.8}])
            self.assertEqual(restored.state['latest'], {'step':1,'score':.8})

    def test_top_three_plus_latest_survive_native_deletion_and_restart(self):
        # Tiny stand-in files test retention policy, not model deserialization.
        import shutil
        from .native_retention import CheckpointRetention, REQUIRED, digest, write_json
        identity = dict(run_id='retention-test', config_hash='c', model_contract_hash='m')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keeper = CheckpointRetention(root, identity)
            for step, score in [(1, .7), (2, .6), (3, .8), (4, .7), (5, .1)]:
                folder = root/f'global_step_{step}'
                folder.mkdir()
                for name in REQUIRED:
                    path = folder/name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'stand-in-for-ranking-check')
                runtime = dict(identity, iteration_id=step, gradus_state={'last_iteration':step})
                write_json(folder/'native_runtime.json', runtime)
                write_json(folder/'NATIVE_COMPLETE.json', dict(runtime,
                    runtime_sha256=digest(folder/'native_runtime.json'), rng_sha256=digest(folder/'native_driver_rng.pt')))
                keeper.record_validation(step, score)
                keeper.preserve(folder, score)
                shutil.rmtree(folder)
            restored = CheckpointRetention(root, identity)
            self.assertEqual([r['step'] for r in restored.state['best']], [3,4,1])
            self.assertEqual(restored.state['latest']['step'], 5)
            for step in (1,3,4,5):
                self.assertTrue((root/'best_checkpoints'/f'global_step_{step}'/'actor/optim_world_size_2_rank_0.pt').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
