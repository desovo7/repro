"""Durable WebShop persistence, adapted from the inspected A800 native run.
Method state joins the manifest before the completion marker is published.
"""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import random
import sys
import zipfile

from .native_retention import CheckpointRetention, REQUIRED, digest, validation_score, write_json


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def stream_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_files(folder):
    manifest = json.loads((folder / 'native_files.json').read_text())
    for name, item in manifest.items():
        p = folder / name
        if p.stat().st_size != item['size'] or stream_digest(p) != item['sha256']:
            raise ValueError('Checkpoint integrity mismatch: ' + name)


def make_worker(native):
    class DurableWebshopWorker(native):
        def __init__(self, seed, env_kwargs):
            import gym
            original = gym.make
            def compatible_make(name, **kwargs):
                kwargs['disable_env_checker'] = True
                return original(name, **kwargs)
            with patch.object(gym, 'make', compatible_make):
                super().__init__(seed, env_kwargs)

        def persistence_state(self):
            state = rng_state()
            # Native collectors reset all sessions before each new rollout.
            # Preserve historical session records as well as every RNG stream.
            state['user_sessions'] = self.env.unwrapped.server.user_sessions
            return state

        def restore_persistence_state(self, state):
            self.env.unwrapped.server.user_sessions = state['user_sessions']
            restore_rng(state)
            return True
    return DurableWebshopWorker


def make_trainer(native):
    class DurableGiGPOTrainer(native):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.identity = dict(self.config.native_persistence.identity)
            self.retention = CheckpointRetention(self.config.trainer.default_local_dir, self.identity,
                self.config.native_persistence.get('diagnostic_steps', []))
            self.last_saved = None
            self.last_validation = None
            self.run_root = Path(self.config.trainer.default_local_dir).parent

        def _validate(self):
            metrics = super()._validate()
            write_json(self.run_root / 'validation' / f'iteration_{self.global_steps}.json',
                       dict(step=self.global_steps, metrics=metrics))
            if self.global_steps > 0 and not self.config.trainer.val_only:
                score = validation_score(metrics)
                self.retention.record_validation(self.global_steps, score)
                self.last_validation = (self.global_steps, score)
                frequency = int(self.config.trainer.save_freq)
                scheduled = frequency > 0 and (self.global_steps % frequency == 0 or
                                               self.global_steps >= self.total_training_steps)
                if not scheduled and self.retention.qualifies(self.global_steps, score):
                    self._save_checkpoint()
            return metrics

        def _save_checkpoint(self):
            import ray
            import torch
            if self.last_saved == self.global_steps:
                return
            folder = Path(self.config.trainer.default_local_dir) / f'global_step_{self.global_steps}'
            self.retention.assert_new_destination(folder)
            driver = rng_state()
            cursors, worker_states = {}, {}
            for name, manager in [('train', self.envs), ('validation', self.val_envs)]:
                kind, keys, pos, gaussian, cached = manager.envs._rng.get_state()
                cursors[name] = dict(kind=kind, keys=keys.tolist(), pos=int(pos),
                                     gaussian=int(gaussian), cached=float(cached))
                worker_states[name] = ray.get([w.persistence_state.remote() for w in manager.envs._workers])
            driver['environment_workers'] = worker_states
            driver['validation_dataloader'] = self.val_dataloader.state_dict()
            result = super()._save_checkpoint()
            runtime = dict(self.identity, iteration_id=self.global_steps,
                           method='GRADUS-WebShop aligned with deployed ALFWorld R2', gradus_state=self.ws_runtime.engine.state_dict(), task_cursors=cursors,
                           boundary='completed training iteration before next collector reset', world_size=2)
            write_json(folder / 'native_runtime.json', runtime)
            temporary = folder / 'native_driver_rng.pt.tmp'
            torch.save(driver, temporary)
            temporary.replace(folder / 'native_driver_rng.pt')
            manifest = {}
            for p in sorted(folder.rglob('*')):
                if not p.is_file():
                    continue
                if p.suffix == '.pt':
                    with zipfile.ZipFile(p) as archive:
                        if not any(n.endswith('/data.pkl') for n in archive.namelist()):
                            raise ValueError('Invalid PyTorch checkpoint archive: ' + str(p))
                manifest[str(p.relative_to(folder))] = dict(size=p.stat().st_size, sha256=stream_digest(p))
            write_json(folder / 'native_files.json', manifest)
            write_json(folder / 'NATIVE_COMPLETE.json', dict(runtime,
                runtime_sha256=digest(folder / 'native_runtime.json'),
                rng_sha256=digest(folder / 'native_driver_rng.pt'),
                checkpoint_manifest_sha256=digest(folder / 'native_files.json')))
            score = self.last_validation[1] if self.last_validation and self.last_validation[0] == self.global_steps else None
            self.retention.preserve(folder, score)
            self.last_saved = self.global_steps
            # Disk I/O and checkpoint RPCs must not advance the training RNG.
            restore_rng(driver)
            print(f'NATIVE_COMPLETE step={self.global_steps} files={len(manifest)} score={score}', flush=True)
            return result

        def _load_checkpoint(self):
            import numpy as np
            import ray
            import torch
            mode, requested = self.config.trainer.resume_mode, self.config.trainer.resume_from_path
            decision = self.retention.resolve_resume(mode, requested)
            selected = decision['selected']
            try:
                self.retention.prepare_native_resume(decision)
                if selected:
                    folder = Path(selected['path'])
                    marker = json.loads((folder / 'NATIVE_COMPLETE.json').read_text())
                    if digest(folder / 'native_files.json') != marker['checkpoint_manifest_sha256']:
                        raise ValueError('Checkpoint file manifest changed')
                    verify_files(folder)
                    self.config.trainer.resume_mode = 'resume_path'
                    self.config.trainer.resume_from_path = str(folder)
                try:
                    result = super()._load_checkpoint()
                finally:
                    self.config.trainer.resume_mode = mode
                    self.config.trainer.resume_from_path = requested
                if selected:
                    if self.global_steps != selected['step']:
                        raise ValueError('Native checkpoint step differs from verified state')
                    runtime = json.loads((folder / 'native_runtime.json').read_text())
                    driver = torch.load(folder / 'native_driver_rng.pt', map_location='cpu', weights_only=False)
                    self.ws_runtime.engine.load_state_dict(runtime['gradus_state'])
                    if self.ws_runtime.engine.last_iteration != self.global_steps:
                        raise ValueError('GRADUS state and model checkpoint differ')
                    for name, manager in [('train', self.envs), ('validation', self.val_envs)]:
                        cursor = runtime['task_cursors'][name]
                        manager.envs._rng.set_state((cursor['kind'], np.asarray(cursor['keys'], dtype=np.uint32),
                                                    cursor['pos'], cursor['gaussian'], cursor['cached']))
                        workers, states = manager.envs._workers, driver['environment_workers'][name]
                        if len(workers) != len(states):
                            raise ValueError('Environment worker count changed on resume')
                        assert all(ray.get([w.restore_persistence_state.remote(s)
                                            for w, s in zip(workers, states, strict=True)]))
                    self.val_dataloader.load_state_dict(driver['validation_dataloader'])
                    restore_rng(driver)
                    self.config.trainer.val_before_train = False
                    print(f'NATIVE_RESUMED step={self.global_steps} integrity_verified=True', flush=True)
                self.retention.finish_resume(decision)
                return result
            except Exception as exc:
                self.retention.finish_resume(decision, error=exc)
                raise
    return DurableGiGPOTrainer

