"""Thin composition around the original trainer and worker group."""
from pathlib import Path

from .runtime import ExternalRuntime
from .common import value


def audit_actor_role_mapping(role_worker_mapping):
    """Select audited Ray workers while keeping their native base importable.

    Rebinding fsdp_workers.ActorRolloutRefWorker makes cloudpickle serialize
    that native class by value. Its inherited methods can then retain a
    different __class__ identity, breaking the original zero-argument super().
    The original TaskRunner has already selected its roles; replace only those
    Ray handles and never the module attribute used to resolve the native base.
    """
    import ray
    from verl.trainer.ppo.ray_trainer import Role
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from .actors import make_audited_actor_worker

    mapping = dict(role_worker_mapping)
    if Role.ActorRollout not in mapping:
        raise ValueError("External RePro requires the original ActorRollout role")
    roles = [role for role in (Role.ActorRollout, Role.RefPolicy) if role in mapping]
    for role in roles:
        selected = mapping[role].__ray_actor_class__
        if ActorRolloutRefWorker not in selected.__mro__:
            raise TypeError(f"Unexpected native worker for {role}: {selected}")
    audited = ray.remote(make_audited_actor_worker(ActorRolloutRefWorker))
    for role in roles:
        mapping[role] = audited
    return mapping


class ActorUpdateProxy:
    def __init__(self, native, runtime):
        self.native = native
        self.runtime = runtime

    def __getattr__(self, name):
        return getattr(self.native, name)

    def update_actor(self, batch):
        self.runtime.inject(batch)
        output = self.native.update_actor(batch)
        self.runtime.commit(output)
        return output


def make_external_trainer(native_trainer):
    class ExternalTrainer(native_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.role_worker_mapping = audit_actor_role_mapping(self.role_worker_mapping)
            self.external_runtime = ExternalRuntime(self.config)
            self.traj_collector.external_runtime = self.external_runtime
            self.traj_collector.external_trainer = self

        def init_workers(self):
            result = super().init_workers()
            self.actor_rollout_wg = ActorUpdateProxy(self.actor_rollout_wg, self.external_runtime)
            return result

        def _save_checkpoint(self):
            root = Path(self.config.trainer.default_local_dir)
            tracker = root / "latest_checkpointed_iteration.txt"
            previous = tracker.read_bytes() if tracker.exists() else None
            result = super()._save_checkpoint()
            folder = root / f"global_step_{self.global_steps}"
            try:
                self.external_runtime.save(folder)
            except Exception:
                # Native saving updates its pointer first. Restore only a known
                # complete retained checkpoint; otherwise retain the incomplete
                # pointer so a resume fails closed instead of silently restarting.
                if previous is not None:
                    prior = root / f"global_step_{previous.decode().strip()}"
                    if (prior / "REPRO_EXTERNAL_COMPLETE.json").is_file() and (prior / "actor").is_dir():
                        temporary = tracker.with_suffix(".txt.tmp")
                        temporary.write_bytes(previous)
                        temporary.replace(tracker)
                raise
            return result

        def _load_checkpoint(self):
            result = super()._load_checkpoint()
            if self.global_steps > 0:
                if value(self.config, "trainer.resume_mode") == "resume_path":
                    folder = Path(self.config.trainer.resume_from_path)
                else:
                    folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
                self.external_runtime.load(folder)
            return result

    return ExternalTrainer

