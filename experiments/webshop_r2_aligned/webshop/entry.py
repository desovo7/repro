"""Launchable candidate entry. Importing this module starts no jobs."""
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import os
import sys

from .core import Config, JVM


def validate_config(config):
    from omegaconf import OmegaConf
    if config.env.env_name != 'Webshop' or config.env.max_steps != 15 or config.env.rollout.n != 8:
        raise ValueError('GRADUS-WebShop requires the native 15-step/group-8 protocol')
    if config.env.history_length != 2 or config.data.train_batch_size != 16 or config.data.val_batch_size != 128:
        raise ValueError('Changing worker counts/history changes this frozen comparison')
    if config.trainer.nnodes != 1 or config.trainer.n_gpus_per_node != 2:
        raise ValueError('Checkpoint and optimizer auditing require two GPUs')
    if config.algorithm.adv_estimator != 'gigpo' or config.algorithm.gigpo.enable_similarity:
        raise ValueError('Expected exact native GiGPO anchors')
    if config.trainer.del_local_ckpt_after_load or set(config.actor_rollout_ref.actor.checkpoint.contents) != {'model', 'optimizer', 'extra'}:
        raise ValueError('Full checkpoint protection required')
    if config.ray_init.runtime_env.env_vars.get('JAVA_TOOL_OPTIONS') != JVM:
        raise ValueError('Ray workers must inherit the capped JVM options')
    Config(**dict(config.gradus_webshop))
    payload = OmegaConf.to_container(config, resolve=False)
    actual = payload['native_persistence']['identity']['config_hash']
    payload['native_persistence']['identity']['config_hash'] = 'pending'
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if actual != expected:
        raise ValueError('Frozen run configuration has changed')


def verify_release(config):
    package = Path(__file__).parent
    run = package.parent
    for root, manifest in ((package, config.webshop_release.files),
                           (run / 'source', config.webshop_release.source),
                           (run / 'data', config.webshop_release.data)):
        for relative, expected in manifest.items():
            path = root / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen code/data changed: ' + str(path))


def make_worker(native):
    from .persistence import make_worker as durable_worker
    class WebshopWorker(durable_worker(native)):
        def __init__(self, seed, env_kwargs):
            if os.environ.get('JAVA_TOOL_OPTIONS') != JVM:
                raise ValueError('Uncapped worker JVM refused')
            self.ws_seed = seed
            super().__init__(seed, env_kwargs)

        def reset(self, idx):
            obs, info = super().reset(idx)
            info = dict(info)
            # Identity for grouping/audit only. Neither field enters features.
            info['ws_identity'] = [self.ws_seed, int(idx)]
            return obs, info
    return WebshopWorker


def run(instance, config):
    from verl.trainer import main_ppo
    import agent_system.multi_turn_rollout as rollout
    from agent_system.environments.env_package.webshop import envs
    from .integration import make_collector, make_trainer
    validate_config(config)
    verify_release(config)
    original = main_ppo.TaskRunner.__ray_metadata__.modified_class.run
    with ExitStack() as stack:
        stack.enter_context(patch.object(envs, 'WebshopWorker', make_worker(envs.WebshopWorker)))
        stack.enter_context(patch.object(rollout, 'TrajectoryCollector', make_collector(rollout.TrajectoryCollector)))
        stack.enter_context(patch.object(main_ppo, 'RayPPOTrainer', make_trainer(main_ppo.RayPPOTrainer)))
        return original(instance, config)


def main():
    import ray
    from omegaconf import OmegaConf
    from verl.trainer import main_ppo
    config = OmegaConf.load(sys.argv[1])
    validate_config(config)
    verify_release(config)
    os.environ['JAVA_TOOL_OPTIONS'] = JVM
    @ray.remote(num_cpus=1)
    class Runner:
        def run(self, config):
            from webshop.entry import run
            return run(self, config)
    try:
        with patch.object(main_ppo, 'TaskRunner', Runner):
            main_ppo.run_ppo(config)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
