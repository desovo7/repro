"""Scope v8.3 hooks to one native TaskRunner; never modify official files."""
from contextlib import contextmanager

from repro_external.common import value
from repro_external.install import validate_external_config


@contextmanager
def install(config):
    validate_external_config(config)
    if value(config,'trainer.nnodes')!=1 or value(config,'trainer.n_gpus_per_node')!=2:
        raise ValueError('v8.3 requires exactly two GPUs on one host')
    if value(config,'repro.enabled',False):raise ValueError('Legacy RePro must not be stacked with v8.3')
    from verl.trainer import main_ppo
    import agent_system.multi_turn_rollout as collector_module
    from agent_system.environments.env_package.alfworld import envs as env_module
    from .collector import make_collector
    from .trainer import make_trainer
    from .environment import make_cursor_worker
    from .traces import preload_task_metadata
    original_get=env_module.get_environment

    def ordered_environment(name):
        native_type=original_get(name)
        def construct(*args,**kwargs):
            result=native_type(*args,**kwargs)
            # Native file-system enumeration differs across hosts even when
            # files match. Both v8.3 arms use the same full canonical order.
            result.game_files=sorted(result.game_files)
            preload_task_metadata(result.game_files,split=kwargs.get('train_eval','unknown'))
            return result
        return construct

    replacements=[(main_ppo,'RayPPOTrainer',make_trainer(main_ppo.RayPPOTrainer)),
                  (collector_module,'TrajectoryCollector',make_collector(collector_module.TrajectoryCollector)),
                  (env_module,'AlfworldWorker',make_cursor_worker(env_module.AlfworldWorker)),
                  (env_module,'get_environment',ordered_environment)]
    originals=[(module,name,getattr(module,name)) for module,name,_ in replacements]
    try:
        for module,name,replacement in replacements:setattr(module,name,replacement)
        yield
    finally:
        for module,name,original in reversed(originals):setattr(module,name,original)
