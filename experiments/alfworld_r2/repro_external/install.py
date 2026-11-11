"""Scoped in-memory external class bindings; official source bytes remain untouched."""
from contextlib import contextmanager

from .common import value


def validate_external_config(config):
    required = {
        "algorithm.adv_estimator": "gigpo",
        "algorithm.progpo.enable": False,
        "algorithm.filter_groups.enable": False,
        "actor_rollout_ref.rollout.mode": "sync",
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 1,
        "trainer.critic_warmup": 0,
        "env.env_name": "alfworld/AlfredTWEnv",
    }
    for key, expected in required.items():
        if value(config, key, expected) != expected:
            raise ValueError(f"External RePro requires {key}={expected!r}")
    if value(config, "algorithm.wm_edbs.wm_model_path"):
        raise ValueError("External RePro cannot combine with WM/EDBS selection")
    if int(value(config, "env.rollout.n", 0)) < 1:
        raise ValueError("External RePro requires grouped original trajectories")
    if int(value(config, "env.max_steps", 0)) < 1:
        raise ValueError("External RePro requires a positive native horizon")


@contextmanager
def install(config):
    validate_external_config(config)
    from verl.trainer import main_ppo
    import agent_system.multi_turn_rollout as collector_module
    from agent_system.environments.env_package.alfworld import envs as environment_module
    from .collector import make_external_collector
    from .environment import make_external_worker
    from .trainer import make_external_trainer

    replacements = [
        (main_ppo, "RayPPOTrainer", make_external_trainer(main_ppo.RayPPOTrainer)),
        (collector_module, "TrajectoryCollector", make_external_collector(collector_module.TrajectoryCollector)),
        (environment_module, "AlfworldWorker", make_external_worker(environment_module.AlfworldWorker,
            max_steps=int(value(config, "env.max_steps")))),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in replacements]
    try:
        for module, name, replacement in replacements:
            setattr(module, name, replacement)
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)

