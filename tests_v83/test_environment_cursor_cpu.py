"""Real full-pool TextWorld CPU checks for the native task-iterator cursor."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch


PROBE_RECORDS = []


def record_event(**record):
    PROBE_RECORDS.append(record)
    print(json.dumps(record), flush=True)


@unittest.skipUnless(os.environ.get('V83_NATIVE_CPU') == '1', 'requires native ALFWorld CPU runtime')
class NativeTaskCursorTests(unittest.TestCase):
    def setUp(self):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('CPU tests must hide CUDA devices')
        from agent_system.environments.env_package.alfworld import envs as native
        from repro_v83.environment import make_cursor_worker
        self.native = native
        self.worker_class = make_cursor_worker(native.AlfworldWorker)
        self.config_path = Path(native.__file__).resolve().parent / 'configs/config_tw.yaml'
        self.workers = []

    def tearDown(self):
        for worker in self.workers:
            worker.env.close()
        import torch
        import ray
        self.assertFalse(torch.cuda.is_initialized())
        self.assertFalse(ray.is_initialized())

    def base(self, split):
        config = self.native.load_config_file(str(self.config_path))
        config['dataset']['num_train_games'] = -1
        config['dataset']['num_eval_games'] = -1
        base = self.native.get_environment(config['env']['type'])(config, train_eval=split)
        self.assertGreater(len(base.game_files), 100)
        return config, base

    def worker(self, config, base, seed=7):
        worker = self.worker_class(config, seed, base)
        self.workers.append(worker)
        return worker

    def test_full_pool_next_games_and_observations_after_restore(self):
        for split in ['train', 'eval_in_distribution', 'eval_out_of_distribution']:
            with self.subTest(split=split):
                config, base = self.base(split)
                original = self.worker(config, base)
                for _ in range(4):
                    original.reset()
                saved = original.v83_task_cursor()
                self.assertEqual(saved['reset_count'], 4)
                restored = self.worker(config, base)
                self.assertEqual(restored.v83_task_cursor()['reset_count'], 0)
                inner = getattr(restored.env, 'unwrapped', restored.env)
                with patch.object(inner, 'reset', side_effect=AssertionError('restore called reset')), \
                        patch.object(inner.batch_env, 'load', side_effect=AssertionError('restore called load')), \
                        patch.object(inner.batch_env, 'reset', side_effect=AssertionError('restore reset game')):
                    self.assertEqual(restored.v83_restore_task_cursor(saved), saved)
                self.assertEqual(getattr(inner, 'last_commands', None), None)
                comparisons = []
                for _ in range(4):
                    left_obs, left_info = original.reset()
                    right_obs, right_info = restored.reset()
                    self.assertEqual(left_info['extra.gamefile'], right_info['extra.gamefile'])
                    self.assertEqual(left_obs, right_obs)
                    self.assertEqual(left_info['admissible_commands'], right_info['admissible_commands'])
                    # Stepping remains delegated; same native action must agree.
                    left_step, right_step = original.step('look'), restored.step('look')
                    self.assertEqual(left_step[0], right_step[0])
                    self.assertEqual(left_step[1], right_step[1])
                    self.assertEqual(left_step[2], right_step[2])
                    comparisons.append(left_info['extra.gamefile'][0])
                self.assertEqual(original.v83_task_cursor(), restored.v83_task_cursor())
                record_event(event='cursor_restore_passed', split=split,
                    full_pool_size=len(base.game_files), saved_reset_count=4,
                    compared_resets=4, compared_steps=4, gamefiles=comparisons,
                    cursor=saved, restore_calls_load_or_reset=False)

    def test_invalid_cursor_fails_before_mutating_iterator(self):
        config, base = self.base('train')
        worker = self.worker(config, base)
        worker.reset()
        saved = worker.v83_task_cursor()
        for key, value in [('schema', 'unknown'), ('seed', 8),
                           ('gamefiles_hash', 'bad'), ('reset_count', -1),
                           ('reset_count', 1.5), ('reset_count', True)]:
            bad = dict(saved, **{key: value})
            with self.assertRaises(ValueError):
                worker.v83_restore_task_cursor(bad)
            self.assertEqual(worker.v83_task_cursor(), saved)
        with self.assertRaises(ValueError):
            worker.v83_restore_task_cursor(dict(saved, extra='not allowed'))
        fresh = self.worker(config, base)
        fresh.v83_restore_task_cursor(saved)
        self.assertEqual(worker.reset()[0], fresh.reset()[0])
        record_event(event='invalid_cursor_rejected', cases=7,
                     rejection_preserves_next_game=True)

    def test_task_pool_order_mismatch_rejected_before_restore(self):
        config, base = self.base('train')
        worker = self.worker(config, base)
        worker.reset()
        saved = worker.v83_task_cursor()
        base.game_files = list(reversed(base.game_files))
        reordered = self.worker(config, base)
        with self.assertRaisesRegex(ValueError, 'ordered task pool mismatch'):
            reordered.v83_restore_task_cursor(saved)
        self.assertEqual(reordered.v83_task_cursor()['reset_count'], 0)
        record_event(event='ordered_pool_mismatch_rejected',
                     full_pool_size=len(base.game_files), restored_reset_count=0)

    def test_production_hook_preloads_complete_sorted_native_pools(self):
        from omegaconf import OmegaConf
        from repro_v83.install import install
        from repro_v83.traces import task_from_path,TASK_PRELOAD_RECORDS
        config = OmegaConf.load(os.environ['V83_NATIVE_CONFIG'])
        task_from_path.cache_clear()
        counts = {'train':3553, 'eval_in_distribution':140, 'eval_out_of_distribution':134}
        original = self.native.get_environment
        with install(config):
            for split, expected in counts.items():
                _, base = self.base(split)
                self.assertEqual(base.game_files, sorted(base.game_files))
                record = TASK_PRELOAD_RECORDS[-1]
                self.assertEqual(record['split'], split)
                self.assertEqual(record['tasks'], expected)
                self.assertEqual(record['new_metadata_reads'], expected)
                before = task_from_path.cache_info()
                tasks = [task_from_path(path) for path in base.game_files]
                after = task_from_path.cache_info()
                self.assertEqual(after.misses, before.misses)
                self.assertEqual(after.hits-before.hits, expected)
                self.assertEqual(len({task.family for task in tasks}), 6)
                record_event(event='production_metadata_preload_passed', **record)
        self.assertIs(self.native.get_environment, original)


if __name__ == '__main__':
    unittest.main()
