"""Composition preserves the official sampler, reward, loss and PPO update."""
from pathlib import Path

from repro_external.common import value,atomic_json
from repro_external.trainer import ActorUpdateProxy,audit_actor_role_mapping
from .runtime import RuntimeV83


def make_trainer(native):
    class TrainerV83(native):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.role_worker_mapping=audit_actor_role_mapping(self.role_worker_mapping)
            self.v83_runtime=RuntimeV83(self.config)
            self.traj_collector.v83_runtime=self.v83_runtime
            self.traj_collector.v83_trainer=self

        def init_workers(self):
            result=super().init_workers()
            self.actor_rollout_wg=ActorUpdateProxy(self.actor_rollout_wg,self.v83_runtime)
            return result

        def _save_checkpoint(self):
            root=Path(self.config.trainer.default_local_dir)
            tracker=root/'latest_checkpointed_iteration.txt'
            previous=tracker.read_bytes() if tracker.exists() else None
            cursors=self._v83_capture_cursors()
            result=super()._save_checkpoint()
            folder=root/f'global_step_{self.global_steps}'
            try:self.v83_runtime.save(folder,task_cursors=cursors)
            except Exception:
                if previous is not None:
                    prior=root/f'global_step_{previous.decode().strip()}'
                    if (prior/'V83_COMPLETE.json').is_file() and (prior/'actor').is_dir():
                        tmp=tracker.with_suffix('.txt.tmp');tmp.write_bytes(previous);tmp.replace(tracker)
                raise
            return result

        def _v83_capture_cursors(self):
            import ray
            return {name:ray.get([worker.v83_task_cursor.remote() for worker in manager.envs.workers])
                    for name,manager in [('train',self.envs),('validation',self.val_envs)]}

        def _v83_restore_cursors(self,cursors):
            import ray
            for name,manager in [('train',self.envs),('validation',self.val_envs)]:
                workers=manager.envs.workers
                if len(workers)!=len(cursors[name]):raise ValueError('Checkpoint environment count mismatch')
                restored=ray.get([worker.v83_restore_task_cursor.remote(saved) for worker,saved in zip(workers,cursors[name])])
                if restored!=cursors[name]:raise ValueError('Task cursor restoration did not match saved state')

        def _load_checkpoint(self):
            result=super()._load_checkpoint()
            if self.global_steps>0:
                folder=Path(self.config.trainer.resume_from_path) if value(self.config,'trainer.resume_mode')=='resume_path' else Path(self.config.trainer.default_local_dir)/f'global_step_{self.global_steps}'
                self.v83_runtime.load(folder)
                self._v83_restore_cursors(self.v83_runtime.task_cursors)
                # The initial validation already occurred in this same run.
                # Repeating it at resume consumes extra samples and advances
                # the validation task iterator beyond an uninterrupted run.
                if not value(self.config,'trainer.val_only',False):
                    self.config.trainer.val_before_train=False
                atomic_json(self.v83_runtime.output/'resume_events'/f'iteration_{self.global_steps}.json',
                    dict(checkpoint=str(folder),restored_environment_cursors=True,
                         duplicate_initial_validation_skipped=not value(self.config,'trainer.val_only',False),
                         note='Task sampling cursor restored; bitwise equality of stochastic GPU generation is not claimed.'))
            return result

    return TrainerV83
