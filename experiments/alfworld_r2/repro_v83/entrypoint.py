"""Invoke the canonical native TaskRunner under process-local v8.3 hooks."""
import inspect
import sys

_NATIVE_TASK_RUNNER=None


def invoke_native_task_runner(instance,config):
    from verl.trainer import main_ppo
    from .install import install
    native=_NATIVE_TASK_RUNNER or main_ppo.TaskRunner
    method=native.__ray_metadata__.modified_class.run
    if inspect.unwrap(method).__globals__ is not vars(main_ppo):raise RuntimeError('Detached native TaskRunner globals')
    with install(config):return method(instance,config)


def run(config):
    global _NATIVE_TASK_RUNNER
    import ray
    from verl.trainer import main_ppo
    native=main_ppo.TaskRunner;previous=_NATIVE_TASK_RUNNER;_NATIVE_TASK_RUNNER=native

    @ray.remote(num_cpus=1)
    class TaskRunnerV83:
        def run(self,config):
            from repro_v83.entrypoint import invoke_native_task_runner
            return invoke_native_task_runner(self,config)

    main_ppo.TaskRunner=TaskRunnerV83
    try:main_ppo.run_ppo(config)
    finally:main_ppo.TaskRunner=native;_NATIVE_TASK_RUNNER=previous


def main():
    from omegaconf import OmegaConf
    if len(sys.argv)!=2:raise SystemExit('usage: python -m repro_v83.entrypoint CONFIG.yaml')
    run(OmegaConf.load(sys.argv[1]))


if __name__=='__main__':main()
