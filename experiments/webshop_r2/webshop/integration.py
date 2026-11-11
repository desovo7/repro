"""Composition around native WebShop collection and the unchanged GiGPO actor."""
from pathlib import Path
from collections import Counter
import json
import math

from .core import Config, CreditEngine, stable_hash, token_injections
from .native_retention import write_json


class Capture:
    def __init__(self, native):
        self.native = native

    def __getattr__(self, name):
        return getattr(self.native, name)

    def reset(self, *args, **kwargs):
        obs, infos = self.native.reset(*args, **kwargs)
        self.initial = list(obs['anchor'])
        self.instructions = list(self.native.tasks)
        self.task_ids = [stable_hash([info['ws_identity'], task]) for info, task in zip(infos, self.instructions, strict=True)]
        self.available = [info['available_actions'] for info in infos]
        self.steps = []
        return obs, infos

    def step(self, text_actions, *args, **kwargs):
        generated = list(text_actions)
        obs, rewards, dones, infos = self.native.step(text_actions, *args, **kwargs)
        records = []
        for i, info in enumerate(infos):
            records.append(dict(observation_after=obs['anchor'][i],
                                action=self.native.memory[i][-1]['action'], generated_text=generated[i],
                                available_before=self.available[i], won=bool(info['won']),
                                format_valid=bool(info['is_action_valid']),
                                environment_done=bool(dones[i]), official_environment_reward=float(rewards[i])))
        self.available = [info['available_actions'] for info in infos]
        self.steps.append(records)
        return obs, rewards, dones, infos


def captured_traces(rows_by_env, capture, run_id, iteration, train):
    traces = []
    for env_index, rows in enumerate(rows_by_env):
        active = [(time, row) for time, row in enumerate(rows) if bool(row['active_masks'])]
        if not active:
            raise ValueError('Missing original trajectory')
        first = active[0][1]
        trace = dict(run_id=run_id, iteration_id=iteration, split='train' if train else 'validation',
                     task_uid=capture.task_ids[env_index], instruction=capture.instructions[env_index],
                     group_uid=str(first['uid']), trajectory_uid=str(first['traj_uid']), steps=[])
        observation = capture.initial[env_index]
        for i, (time, row) in enumerate(active):
            if str(row['uid']) != trace['group_uid'] or str(row['traj_uid']) != trace['trajectory_uid']:
                raise ValueError('Native identity changes within trajectory')
            if row['anchor_obs'] != observation:
                raise ValueError('Native anchor and captured observation disagree')
            uid = f'{run_id}:{iteration}:{trace["trajectory_uid"]}:{i}'
            row['ws_action_uid'] = uid
            record = dict(capture.steps[time][env_index], observation_before=observation, step_id=i, action_uid=uid)
            if float(row['rewards']) != record['official_environment_reward']:
                raise ValueError('Native reward capture mismatch')
            trace['steps'].append(record)
            observation = record['observation_after']
        trace['success'] = trace['steps'][-1]['won']
        traces.append(trace)
    return traces


def make_collector(native):
    class WebshopCollector(native):
        ws_runtime = None

        def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=True):
            kwargs = dict(gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, envs=envs, is_train=is_train)
            if self.ws_runtime is None:
                return super().multi_turn_loop(**kwargs)
            if self.config.env.max_steps != 15 or self.config.algorithm.filter_groups.enable:
                raise ValueError('Expected native vanilla collection with a 15-step horizon')
            train = bool(is_train)
            iteration = self.ws_trainer.global_steps
            capture = Capture(envs)
            self._ws_pending = (capture, iteration, train)
            self._ws_gathered = False
            try:
                result = super().multi_turn_loop(**dict(kwargs, envs=capture))
                if not self._ws_gathered:
                    raise ValueError('Native loop returned without gathering captured trajectories')
                return result
            finally:
                self._ws_pending = None

        def gather_rollout_data(self, total_batch_list, *args, **kwargs):
            rows = total_batch_list  # exact native keyword contract
            pending = getattr(self, '_ws_pending', None)
            if pending:
                if getattr(self, '_ws_gathered', False):
                    raise ValueError('Repeated native gather within one rollout')
                capture, iteration, train = pending
                if len(capture.steps) > 15:
                    raise ValueError('Native collector exceeded the configured horizon')
                traces = captured_traces(rows, capture, self.ws_runtime.run_id, iteration, train)
                if train:
                    self.ws_runtime.prepare(traces, iteration)
                else:
                    self.ws_runtime.record_validation(traces, iteration)
            # Never replace anchor_obs, native rewards or native returns.
            batch = super().gather_rollout_data(rows, *args, **kwargs)
            if pending:
                expected = {r['ws_action_uid']: (r['anchor_obs'], float(r['rewards']))
                            for trajectory in rows for r in trajectory if bool(r['active_masks'])}
                ids = list(batch.non_tensor_batch['ws_action_uid'])
                if len(ids) != len(expected) or set(ids) != set(expected):
                    raise ValueError('Native gather dropped or duplicated original action identities')
                for uid, anchor, reward in zip(ids, batch.non_tensor_batch['anchor_obs'],
                                              batch.non_tensor_batch['rewards'], strict=True):
                    if expected[uid] != (anchor, float(reward)):
                        raise ValueError('Native gather changed action/anchor/reward alignment')
                self._ws_gathered = True
            return batch
    return WebshopCollector


class Runtime:
    def __init__(self, config):
        self.config = config
        self.run_id = config.native_persistence.identity.run_id
        self.output = Path(config.trainer.default_local_dir).parent / 'gradus'
        self.output.mkdir(parents=True, exist_ok=True)
        self.engine = CreditEngine(Config(**dict(config.gradus_webshop)))
        self.phase = 'idle'

    def prepare(self, traces, iteration):
        if self.phase not in {'idle', 'committed'} or any(t['run_id'] != self.run_id for t in traces):
            raise ValueError('Uncommitted or foreign collection')
        self.result = self.engine.prepare(traces, iteration)
        self.iteration = iteration
        self.phase = 'collected'
        with (self.output / 'train_traces.jsonl').open('a') as f:
            for trace in traces:
                f.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + '\n')

    def record_validation(self, traces, iteration):
        # Validation is audit-only: no parse, prepare, commit or vocabulary update.
        folder = self.output / 'validation_traces'
        folder.mkdir(exist_ok=True)
        with (folder / f'iteration_{iteration}.jsonl').open('a') as f:
            for trace in traces:
                f.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + '\n')

    def inject(self, batch):
        import torch
        if self.phase != 'collected' or batch.meta_info.get('ws_credit_applied'):
            raise ValueError('Duplicate or out-of-order credit')
        base, mask = batch.batch['advantages'], batch.batch['response_mask']
        if base.ndim != 2 or base.shape != mask.shape or not torch.isfinite(base).all() or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('Invalid native advantages or response mask')
        uids = list(batch.non_tensor_batch['ws_action_uid'])
        if len(uids) != base.shape[0]:
            raise ValueError('Actor row identities mismatch')
        weights = mask.sum(-1).detach().cpu().tolist()
        values, self.injection = token_injections(self.result, uids, weights, self.engine.config)
        delta = torch.tensor(values, device=base.device, dtype=base.dtype)[:, None] * mask
        combined = base + delta
        if not torch.isfinite(combined).all():
            raise FloatingPointError('Nonfinite injected advantage')
        batch.batch['advantages'] = combined.detach()
        batch.batch['repro_aux_delta'] = (combined - base).detach()
        batch.meta_info['ws_credit_applied'] = True
        batch.meta_info['repro_external_audit_context'] = dict(output_dir=str(self.output), run_id=self.run_id,
            iteration_id=self.iteration, snapshot_id=f'{self.run_id}:{self.iteration}')
        self.injection['actor_rows'] = len(uids)
        self.injection['unique_actor_actions'] = len(set(uids))
        self.phase = 'injected'

    def commit(self, output):
        if self.phase != 'injected':
            raise ValueError('Actor commit out of order')
        audit = output.meta_info.get('repro_external_audit', {})
        if (audit.get('run_id') != self.run_id or audit.get('iteration_id') != self.iteration
            or audit.get('snapshot_id') != f'{self.run_id}:{self.iteration}'
            or audit.get('world_size') != 2 or not audit.get('all_ranks_finite')
            or audit.get('actual_steps_min', 0) <= 0
            or len({audit.get(k) for k in ('actual_steps_min', 'actual_steps_max', 'expected_steps_min', 'expected_steps_max')}) != 1):
            raise ValueError('Both actor ranks must finish their expected update')
        self.engine.commit(self.iteration)
        metrics = {**self.result['diagnostics'], **self.injection}
        output.meta_info.setdefault('metrics', {}).update({'gradus_ws/' + k: v for k, v in metrics.items()})
        write_json(self.output / 'diagnostics' / f'iteration_{self.iteration}.json', dict(metrics=metrics, audit=audit))
        self.phase = 'committed'


class ActorProxy:
    def __init__(self, native, runtime):
        self.native, self.runtime = native, runtime

    def __getattr__(self, name):
        return getattr(self.native, name)

    def update_actor(self, batch):
        self.runtime.inject(batch)
        result = self.native.update_actor(batch)
        self.runtime.commit(result)
        return result


def audited_roles(mapping):
    import ray
    from verl.trainer.ppo.ray_trainer import Role
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from .actor_audit import make_audited_actor_worker
    mapping = dict(mapping)
    if Role.ActorRollout not in mapping:
        raise ValueError('Expected the native ActorRollout role')
    audited = ray.remote(make_audited_actor_worker(ActorRolloutRefWorker))
    for role in (Role.ActorRollout, Role.RefPolicy):
        if role in mapping:
            if ActorRolloutRefWorker not in mapping[role].__ray_actor_class__.__mro__:
                raise ValueError('Unexpected actor worker class')
            mapping[role] = audited
    return mapping


def make_trainer(native):
    from .persistence import make_trainer as durable_trainer
    class WebshopTrainer(durable_trainer(native)):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ws_runtime = Runtime(self.config)
            self.role_worker_mapping = audited_roles(self.role_worker_mapping)
            self.traj_collector.ws_runtime = self.ws_runtime
            self.traj_collector.ws_trainer = self

        def init_workers(self):
            result = super().init_workers()
            self.actor_rollout_wg = ActorProxy(self.actor_rollout_wg, self.ws_runtime)
            return result

        def _save_checkpoint(self):
            if self.ws_runtime.engine.last_iteration != self.global_steps or self.ws_runtime.phase != 'committed':
                raise ValueError('Checkpoint requires completed method and actor state')
            return super()._save_checkpoint()
    return WebshopTrainer
