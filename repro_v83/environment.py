"""Checkpoint only the native ALFWorld task iterator, without replaying games."""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from pathlib import Path


SCHEMA = 'repro_v83_task_cursor_v1'


def _relative_gamefiles(gamefiles):
    result = []
    for gamefile in gamefiles:
        parts = Path(gamefile).parts
        markers = [i for i, part in enumerate(parts)
                   if part in {'train', 'valid_seen', 'valid_unseen'}]
        if not markers:
            raise ValueError('Cannot establish a portable ALFWorld task-file identity')
        relative = '/'.join(parts[markers[-1]:])
        if len(parts) - markers[-1] != 4 or parts[-1] != 'game.tw-pddl':
            raise ValueError('Unexpected ALFWorld task-file layout')
        result.append(relative)
    if not result or len(set(result)) != len(result):
        raise ValueError('Task pool must be nonempty with unique task-file identities')
    return result


def _ordered_hash(gamefiles):
    content = json.dumps(_relative_gamefiles(gamefiles), separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def make_cursor_worker(native_worker):
    class CursorWorkerV83(native_worker):
        def __init__(self, config, seed, base_env):
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise ValueError('Native task seed must be an integer')
            self._v83_cursor_seed = int(seed)
            self._v83_reset_count = 0
            self._v83_gamefiles_hash = _ordered_hash(base_env.game_files)
            super().__init__(config, seed, base_env)
            self._v83_task_iterator()

        def _v83_task_iterator(self):
            # Pinned textworld.gym.make returns the batch environment directly.
            # Gym wrappers may expose that same type through public unwrapped;
            # never guess a private wrapper chain or call reset to inspect it.
            from textworld.gym.envs.textworld_batch import TextworldBatchGymEnv
            native_env = (self.env if isinstance(self.env, TextworldBatchGymEnv)
                          else getattr(self.env, 'unwrapped', None))
            if not isinstance(native_env, TextworldBatchGymEnv):
                raise TypeError('Unsupported native environment for task cursor restoration')
            if native_env.batch_size != 1:
                raise ValueError('The native ALFWorld worker must own one environment')
            if _ordered_hash(native_env.gamefiles) != self._v83_gamefiles_hash:
                raise ValueError('Native task pool order changed after worker construction')
            iterator = getattr(native_env, '_gamefiles_iterator', None)
            if iterator is None or not hasattr(iterator, '__next__'):
                raise TypeError('Pinned TextWorld task iterator contract is missing')
            return native_env

        def reset(self, *args, **kwargs):
            result = super().reset(*args, **kwargs)
            self._v83_reset_count += 1
            return result

        def v83_task_cursor(self):
            self._v83_task_iterator()
            return dict(schema=SCHEMA, seed=self._v83_cursor_seed,
                        reset_count=self._v83_reset_count,
                        gamefiles_hash=self._v83_gamefiles_hash)

        def v83_restore_task_cursor(self, saved):
            if not isinstance(saved, dict) or set(saved) != {
                    'schema', 'seed', 'reset_count', 'gamefiles_hash'}:
                raise ValueError('Invalid native task cursor schema')
            if saved['schema'] != SCHEMA:
                raise ValueError('Unsupported native task cursor schema version')
            if type(saved['seed']) is not int or saved['seed'] != self._v83_cursor_seed:
                raise ValueError('Native task cursor seed mismatch')
            if saved['gamefiles_hash'] != self._v83_gamefiles_hash:
                raise ValueError('Native task cursor ordered task pool mismatch')
            if type(saved['reset_count']) is not int or saved['reset_count'] < 0:
                raise ValueError('Native task reset count must be a nonnegative integer')
            native_env = self._v83_task_iterator()
            # TextworldBatchGymEnv.seed recreates the seeded shuffled_cycle;
            # native reset consumes batch_size entries before load/reset.
            self.env.seed(self._v83_cursor_seed)
            native_env = self._v83_task_iterator()
            for _ in range(saved['reset_count'] * native_env.batch_size):
                next(native_env._gamefiles_iterator)
            self._v83_reset_count = saved['reset_count']
            return self.v83_task_cursor()

    return CursorWorkerV83
