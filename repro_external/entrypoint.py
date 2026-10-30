"""Run the untouched official TaskRunner with external RePro class composition."""
from __future__ import annotations

import inspect
import sys

_NATIVE_TASK_RUNNER = None


def invoke_native_task_runner(instance, config):
    from verl.trainer import main_ppo
    from .install import install
    # A fresh Ray worker imports the pristine module. Keeping the driver-side
    # original also makes local/Ray-local-mode delegation nonrecursive.
    native_runner = _NATIVE_TASK_RUNNER or main_ppo.TaskRunner
    native_method = native_runner.__ray_metadata__.modified_class.run
    if inspect.unwrap(native_method).__globals__ is not vars(main_ppo):
        raise RuntimeError("Native TaskRunner globals must be the canonical worker module")
    with install(config):
        return native_method(instance, config)


def run(config):
    global _NATIVE_TASK_RUNNER
    import ray
    from verl.trainer import main_ppo

    native_runner = main_ppo.TaskRunner
    previous_runner = _NATIVE_TASK_RUNNER
    _NATIVE_TASK_RUNNER = native_runner

    @ray.remote(num_cpus=1)
    class ExternalTaskRunner:
        def run(self, config):
            # Import by name in the receiver. With ``python -m``, capturing a
            # __main__ helper also serializes the driver's native Ray handle and
            # a detached globals dictionary, silently bypassing scoped bindings.
            from repro_external.entrypoint import invoke_native_task_runner
            # The original function performs model/tokenizer loading, dataset
            # construction, environment construction, worker setup and fit.
            return invoke_native_task_runner(self, config)

    main_ppo.TaskRunner = ExternalTaskRunner
    try:
        # Keep native Ray initialization/runtime-env and synchronous execution.
        main_ppo.run_ppo(config)
    finally:
        main_ppo.TaskRunner = native_runner
        _NATIVE_TASK_RUNNER = previous_runner


def main(argv=None):
    from omegaconf import OmegaConf
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: python -m repro_external.entrypoint RESOLVED_CONFIG.yaml")
    run(OmegaConf.load(argv[0]))


if __name__ == "__main__":
    main()
