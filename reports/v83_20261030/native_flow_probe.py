"""Actual native collector and actual Ray TaskRunner CPU boundary evidence."""
import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import traceback


EVIDENCE={}


class CpuWorkerBoundaryReached(RuntimeError):
    pass


def worker_setup():
    """Only replace environment creation and the GPU-init boundary."""
    from types import SimpleNamespace
    import ray
    import torch
    import gc
    from verl.trainer import main_ppo
    import agent_system.environments as environments
    import agent_system.multi_turn_rollout as collector_module
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer,Role
    import repro_v83.install as installer
    from repro_v83.runtime import RuntimeV83
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    EVIDENCE.update(pid=os.getpid(),assigned_gpu_ids=ray.get_gpu_ids(),
                    cuda_initialized_before=torch.cuda.is_initialized(),
                    gc_freeze_count_before=gc.get_freeze_count())
    original_init=RayPPOTrainer.__init__
    original_install=installer.install
    native_trainer=main_ppo.RayPPOTrainer
    native_collector=collector_module.TrajectoryCollector
    environments.make_envs=lambda config:(SimpleNamespace(cpu_probe_skipped_environment=True),
                                           SimpleNamespace(cpu_probe_skipped_environment=True))

    @contextmanager
    def recording_install(config):
        frames=inspect.stack()
        frame=next(x.frame for x in frames if x.function=='invoke_native_task_runner'
                   and x.frame.f_globals.get('__name__')=='repro_v83.entrypoint')
        native_method=frame.f_locals['method']
        globals_=inspect.unwrap(native_method).__globals__
        del frame,frames
        try:
            with original_install(config):
                EVIDENCE.update(native_globals_are_canonical=globals_ is vars(main_ppo),
                    native_globals_trainer_is_installed=globals_['RayPPOTrainer'] is main_ppo.RayPPOTrainer,
                    installed_trainer=main_ppo.RayPPOTrainer.__qualname__,
                    installed_collector=collector_module.TrajectoryCollector.__qualname__,
                    native_trainer_init_unmodified=RayPPOTrainer.__init__ is original_init)
                yield
        finally:
            EVIDENCE.update(trainer_binding_restored=main_ppo.RayPPOTrainer is native_trainer,
                collector_binding_restored=collector_module.TrajectoryCollector is native_collector,
                gc_restored_after_fit=gc.get_freeze_count()==EVIDENCE['gc_freeze_count_before'],
                automatic_gc_enabled_after_fit=gc.isenabled())
            Path(os.environ['V83_NATIVE_WORKER_REPORT']).write_text(json.dumps(EVIDENCE,indent=2)+'\n')
    installer.install=recording_install

    def stop_at_workers(self):
        runtime=getattr(self,'v83_runtime',None)
        roles={role.name:self.role_worker_mapping[role].__ray_actor_class__ for role in [Role.ActorRollout,Role.RefPolicy]}
        EVIDENCE.update(actual_native_taskrunner_on_stack=any(x.filename.endswith('verl/trainer/main_ppo.py') and x.function=='run'
            for x in inspect.stack()),actual_trainer_class=type(self).__qualname__,
            has_real_v83_runtime=isinstance(runtime,RuntimeV83),
            collector_runtime_connected=self.traj_collector.v83_runtime is runtime,
            collector_trainer_connected=self.traj_collector.v83_trainer is self,
            actual_collector_class=type(self.traj_collector).__qualname__,
            tokenizer_class=type(self.tokenizer).__name__,train_dataset_class=type(self.train_dataset).__name__,
            train_dataset_rows=len(self.train_dataset),val_dataset_rows=len(self.val_dataset),
            actual_train_dataloader_class=type(self.train_dataloader).__name__,
            native_trainer_init_unmodified=RayPPOTrainer.__init__ is original_init,
            audited_actor_roles={name:any(b.__name__=='AuditedActorWorker' for b in cls.__mro__) for name,cls in roles.items()},
            cuda_initialized_after=torch.cuda.is_initialized(),distributed_initialized=torch.distributed.is_initialized())
        # Supply only the boundary handle expected by the wrapper. No native
        # worker creation, model loading or optimizer call is performed.
        self.actor_rollout_wg=SimpleNamespace(cpu_probe_skipped_workers=True)
    RayPPOTrainer.init_workers=stop_at_workers

    def stop_at_fit(self):
        EVIDENCE.update(gc_scope_active_at_fit=gc.get_freeze_count()>0,
            gc_unmodified_at_fit=gc.get_freeze_count()==EVIDENCE['gc_freeze_count_before'],
            automatic_gc_enabled_at_fit=gc.isenabled(),
            actor_proxy_installed=type(self.actor_rollout_wg).__name__=='ActorUpdateProxy',
            native_training_loop_skipped=True)
        raise CpuWorkerBoundaryReached('Native trainer constructed; stopped before native training loop')
    RayPPOTrainer.fit=stop_at_fit


def run_ray_probe(config, scratch):
    import ray
    import torch
    from omegaconf import OmegaConf
    scratch=Path(scratch);scratch.mkdir(parents=True,exist_ok=True)
    worker_report=scratch/'worker_report.json'
    config.repro_v83.output_dir=str(scratch/'runtime')
    cfg=scratch/'resolved.yaml';OmegaConf.save(config,cfg)
    caught=None
    probe_temp=str(config.ray_init._temp_dir)+'-cpu'
    ray.init(address='local',num_cpus=2,num_gpus=0,include_dashboard=False,
        object_store_memory=100*1024*1024,_temp_dir=probe_temp,
        runtime_env={'worker_process_setup_hook':worker_setup,
            'env_vars':{'V83_NATIVE_WORKER_REPORT':str(worker_report),'CUDA_VISIBLE_DEVICES':''}})
    try:
        sys.argv=['repro_v83.entrypoint',str(cfg)]
        runpy.run_module('repro_v83.entrypoint',run_name='__main__')
    except Exception as error:
        caught=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        ray.shutdown()
    result=json.loads(worker_report.read_text()) if worker_report.exists() else {'worker_report_missing':True}
    result.update(caught=caught,driver_pid=os.getpid(),driver_cuda_initialized=torch.cuda.is_initialized(),
        ray_shutdown=not ray.is_initialized(),ray_temp_dir=probe_temp,
        production_ray_temp_dir=str(config.ray_init._temp_dir),
        scope='Real native TaskRunner/trainer/collector/datasets/tokenizer with default GC; no model/GPU/optimizer execution',
        patched_boundaries=['make_envs returns inert objects','native init_workers supplies an inert handle without resource/GPU initialization','native fit stops before the training loop'])
    required=['native_globals_are_canonical','native_globals_trainer_is_installed','native_trainer_init_unmodified',
        'actual_native_taskrunner_on_stack','has_real_v83_runtime','collector_runtime_connected','collector_trainer_connected',
        'trainer_binding_restored','collector_binding_restored','ray_shutdown',
        'gc_unmodified_at_fit','automatic_gc_enabled_at_fit','gc_restored_after_fit',
        'automatic_gc_enabled_after_fit','actor_proxy_installed','native_training_loop_skipped']
    result['status']='passed' if (all(result.get(k) for k in required) and result.get('pid')!=os.getpid()
        and not result.get('assigned_gpu_ids') and not result.get('cuda_initialized_after')
        and not result.get('driver_cuda_initialized') and not result.get('distributed_initialized')
        and result.get('train_dataset_rows')==16 and result.get('val_dataset_rows')==128
        and all(result.get('audited_actor_roles',{}).values())
        and caught and 'CpuWorkerBoundaryReached' in caught['message']) else 'failed'
    return result


def main(args):
    from omegaconf import OmegaConf
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    config=OmegaConf.load(args.config)
    config.repro.enabled=False
    OmegaConf.update(config,'repro_v83',dict(run_id='v83-native-cpu-flow',config_hash='cpu-frozen-contract',
        model_contract_hash='qwen25-native-tokenizer-cpu-probe',output_dir=str(Path(args.scratch)/'collector'),
        enable_tier1a=True,enable_tier1b=args.tier!='1a',enable_tier2=args.tier=='2'),force_add=True)
    from tests_v83.test_native_collector_cpu import run_collector_probe
    result={'collector':run_collector_probe(config,args.compact,Path(args.scratch)/'collector')}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    result['ray_entrypoint']=run_ray_probe(config,Path(args.scratch)/'ray')
    root=Path(__file__).resolve().parents[2]
    result['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*.py')}
    result['config_sha256']=hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]['status'] for k in ['collector','ray_entrypoint']}),flush=True)
    assert result['ray_entrypoint']['status']=='passed', result['ray_entrypoint']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--compact',required=True)
    parser.add_argument('--scratch',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--tier',choices=['1a','1b','2'],default='2')
    args=parser.parse_args()
    # Keep the worker setup hook importable by its module name in Ray children.
    from native_flow_probe import main
    main(args)
