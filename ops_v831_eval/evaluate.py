"""Full-catalog evaluation: one native ALFWorld episode for every frozen task."""
import argparse
import collections
import hashlib
import json
import importlib.metadata
import inspect
import os
import socket
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    from contract import gpu_inventory, read, validate_ready
    card = validate_ready(root, check_environment=True, check_gpus=not args.preflight)
    if not args.preflight:
        attempt = read(root / 'evaluation_attempt.json')
        if (attempt['attempt_id'] != os.environ.get('V831_EVALUATION_ATTEMPT_ID')
                or attempt['run_id'] != card['training_run_id'] or (root / 'execution.json').exists()):
            raise ValueError('Start each fresh evaluation once through run.py')
        current = gpu_inventory(card['resources']['physical_gpu_indices'], card['execution_host'], require_idle=True)
        save(root / 'execution.json', dict(attempt_id=attempt['attempt_id'],
            run_id=card['training_run_id'], execution_host=card['execution_host'],
            hostname=socket.gethostname(), tensor_parallel_size=2,
            physical_gpu_indices=card['resources']['physical_gpu_indices'],
            gpu_inventory={str(k): v for k, v in current.items()},
            cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from safetensors import safe_open
    from verl import DataProto
    from agent_system.multi_turn_rollout import TrajectoryCollector
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager
    from agent_system.environments.env_package.alfworld import alfworld_projection
    from agent_system.environments.env_package.alfworld import envs as alf_native
    from fixed_tasks import FixedTaskEnvs
    for module_path in [inspect.getfile(TrajectoryCollector), alf_native.__file__]:
        assert Path(module_path).resolve().is_relative_to(Path(card['source_root']).resolve())

    config = OmegaConf.load(root / 'training_config.yaml')
    assert config.env.max_steps == 50 and config.env.history_length == 2
    assert config.data.val_batch_size == 32
    assert config.data.max_prompt_length == 2048 and config.data.max_response_length == 512
    assert config.env.seed == 0
    assert config.actor_rollout_ref.rollout.val_kwargs.temperature == 0.4
    assert config.actor_rollout_ref.rollout.val_kwargs.do_sample
    assert config.actor_rollout_ref.rollout.val_kwargs.top_p == 1
    assert config.actor_rollout_ref.rollout.val_kwargs.top_k == -1
    tokenizer = AutoTokenizer.from_pretrained(card['tokenizer_path'], trust_remote_code=False)
    captured = AutoTokenizer.from_pretrained(root / 'checkpoint_actor', trust_remote_code=False)
    assert tokenizer.get_vocab() == captured.get_vocab()
    assert tokenizer.chat_template == captured.chat_template
    assert tokenizer.eos_token_id == captured.eos_token_id
    assert tokenizer.pad_token_id == captured.pad_token_id
    manifest = json.loads((root / 'task_manifest.json').read_text())
    assert {k: len(v['tasks']) for k, v in manifest['splits'].items()} == {'seen': 140, 'unseen': 134}
    for split, data in manifest['splits'].items():
        assert len({t['game_id'] for t in data['tasks']}) == len(data['tasks'])
        assert [t['game_id'] for t in data['tasks']] == sorted(t['game_id'] for t in data['tasks'])
        for task in data['tasks']:
            assert sha(task['gamefile']) == task['sha256'], task['game_id']
    proof = dict(checkpoint=150, tokenizer_vocab_equal=True, chat_template_equal=True,
                 chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                 merged_model_sha256=sha(root / 'hf_model/model.safetensors'),
                 training_config_sha256=sha(root / 'training_config.yaml'),
                 task_manifest_sha256=sha(root / 'task_manifest.json'),
                 all_274_task_files_verified=True)
    proof['versions'] = {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'vllm', 'ray']}
    with safe_open(root / 'hf_model/model.safetensors', framework='pt', device='cpu') as f:
        proof['tensor_count'] = len(f.keys())
        proof['tensor_dtypes'] = sorted({str(f.get_slice(k).get_dtype()) for k in f.keys()})
        key = 'model.layers.0.self_attn.q_proj.weight'
        checkpoint_tensor = f.get_tensor(key)
        assert torch.isfinite(checkpoint_tensor).all()
        proof['sample_weight_shape'] = list(checkpoint_tensor.shape)
    with safe_open(Path(card['tokenizer_path']) / 'model.safetensors', framework='pt', device='cpu') as f:
        base_tensor = f.get_tensor(key).to(checkpoint_tensor.dtype)
        proof['sample_weight_differs_from_base'] = not torch.equal(base_tensor, checkpoint_tensor)
        assert proof['sample_weight_differs_from_base']
    save(root / 'preflight.json', proof)
    print(json.dumps({'event': 'preflight_ok', **proof}), flush=True)
    if args.preflight:
        return

    import ray
    from vllm import LLM, SamplingParams
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
    assert Path(inspect.getfile(vLLMRollout)).resolve().is_relative_to(Path(card['source_root']).resolve())

    # Only engine initialization changes. Generation, padding, EOS and sampling
    # overrides are inherited unmodified from the official evaluation rollout.
    class EvaluationRollout(vLLMRollout):
        world_size = 1

        def __init__(self):
            self.config = config.actor_rollout_ref.rollout
            self.pad_token_id = tokenizer.pad_token_id
            self.lora_kwargs = {}
            self.inference_engine = LLM(
                model=str(root / 'hf_model'), tokenizer=card['tokenizer_path'],
                tensor_parallel_size=card['resources']['tensor_parallel_size'], distributed_executor_backend='mp',
                dtype='bfloat16', enforce_eager=False,
                gpu_memory_utilization=card['gpu_memory_utilization'],
                disable_custom_all_reduce=True, max_model_len=2560,
                max_num_seqs=32, load_format='safetensors', disable_log_stats=True,
                max_num_batched_tokens=8192, enable_chunked_prefill=False,
                enable_prefix_caching=True, trust_remote_code=False, seed=0,
            )
            kwargs = dict(n=1, logprobs=0, max_tokens=512, detokenize=False)
            for k in self.config:
                if hasattr(SamplingParams(), str(k)):
                    kwargs[k] = self.config.get(k)
            self.sampling_params = SamplingParams(**kwargs)

    start = time.time()
    rollout = EvaluationRollout()
    ray.init(address='local', num_cpus=8, num_gpus=0,
             object_store_memory=1 << 30, include_dashboard=False,
             _temp_dir=card['ray_temp_dir'], log_to_driver=False)
    collector = TrajectoryCollector(config, tokenizer)

    class RecordedEnvironment(AlfWorldEnvironmentManager):
        def reset(self, kwargs):
            result = super().reset(kwargs)
            self.rounds = 0
            self.finished = np.zeros(len(self.gamefile), dtype=bool)
            return result

        def step(self, text_actions):
            result = super().step(text_actions)
            self.rounds += 1
            self.finished |= np.asarray(result[2]).reshape(-1).astype(bool)
            if self.rounds % 5 == 0:
                progress = dict(event='step_progress', **self.batch_label,
                    step=self.rounds, batch_size=len(self.finished), finished=int(self.finished.sum()))
                save(root / 'step_progress.json', progress)
                print(json.dumps(progress), flush=True)
            return result

        def success_evaluator(self, *a, **kw):
            self.last_infos = kw['total_infos']
            return super().success_evaluator(*a, **kw)

    assert EvaluationRollout.generate_sequences is vLLMRollout.generate_sequences
    config_path = str(Path(card['source_root']) / 'agent_system/environments/env_package/alfworld/configs/config_tw.yaml')
    aggregate = dict(checkpoint=150, status='running', protocol=card['protocol'], splits={})
    try:
        for split, native_split in card['splits'].items():
            config.env.alfworld.eval_dataset = native_split
            native_config = alf_native.load_config_file(config_path)
            base_env = alf_native.get_environment(native_config['env']['type'])(native_config, train_eval=native_split)
            tasks = manifest['splits'][split]['tasks']
            assert set(base_env.game_files) == {t['gamefile'] for t in tasks}
            backend = FixedTaskEnvs(base_env)
            env = RecordedEnvironment(backend, alfworld_projection, config)
            episodes = []
            try:
                for batch_index, start_index in enumerate(range(0, len(tasks), 32)):
                    batch_tasks = tasks[start_index:start_index + 32]
                    n = len(batch_tasks)
                    backend.assign(batch_tasks)
                    env.batch_label = dict(split=split, batch_index=batch_index, completed_before=len(episodes))
                    # Placeholder validation prompts are replaced by the native
                    # collector with actual environment observations on every step.
                    raw = np.empty(n, dtype=object)
                    raw[:] = [[{'role': 'user', 'content': ''}] for _ in range(n)]
                    gen_batch = DataProto.from_single_dict(dict(
                        input_ids=torch.zeros((n, 1), dtype=torch.long),
                        raw_prompt=raw, data_source=np.array(['alfworld'] * n, dtype=object)))
                    gen_batch.meta_info = dict(eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id, recompute_log_prob=False,
                        do_sample=True, validate=True)
                    batch_t0 = time.time()
                    rows, rewards, lengths, success, uids, _ = collector.vanilla_multi_turn_loop(
                        gen_batch=gen_batch, actor_rollout_wg=rollout, envs=env)
                    assert len(rows) == n and len(success['success_rate']) == n
                    # Native evaluator returns the last ACTIVE transition's won.
                    # Read that same value from recorded info by wrapping the
                    # evaluator, below; this independently checks the count.
                    batch_episodes = []
                    for i, trajectory in enumerate(rows):
                        indices = [j for j, r in enumerate(trajectory) if bool(r['active_masks'])]
                        active = [trajectory[j] for j in indices]
                        active_infos = [env.last_infos[i][j] for j in indices]
                        won = bool(success['success_rate'][i])
                        assert won == bool(active_infos[-1]['won'])
                        assert int(won) == int(rewards[i] > 0)
                        gamefile = env.gamefile[i]
                        task = batch_tasks[i]
                        assert gamefile == task['gamefile'], (gamefile, task)
                        assert native_split.replace('eval_in_distribution', 'valid_seen').replace('eval_out_of_distribution', 'valid_unseen') in gamefile
                        record = dict(split=split, batch_index=batch_index, env_index=i,
                            episode_id=str(uids[i]), gamefile=gamefile, won=won,
                            game_id=task['game_id'], task_family=task['task_family'],
                            manifest_index=task['manifest_index'],
                            steps=len(active), episode_reward=float(rewards[i]),
                            trajectory=[dict(step=j + 1, observation=str(row['anchor_obs']),
                                response=tokenizer.decode(row['responses'], skip_special_tokens=True),
                                reward=float(row['rewards']),
                                won=bool(active_infos[j]['won']),
                                format_valid=bool(row['is_action_valid'])) for j, row in enumerate(active)])
                        assert record['steps'] == int(lengths[i])
                        batch_episodes.append(record)
                    assert sum(r['won'] for r in batch_episodes) == int(success['success_rate'].sum())
                    out = root / f'{split}_batch_{batch_index}.jsonl'
                    with out.open('x') as f:
                        for record in batch_episodes:
                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        f.flush()
                        os.fsync(f.fileno())
                    episodes.extend(batch_episodes)
                    stats = dict(n=len(episodes), successes=sum(r['won'] for r in episodes),
                        unique_games=len({r['gamefile'] for r in episodes}),
                        expected_tasks=len(tasks),
                        batch_seconds=time.time() - batch_t0)
                    stats['success_rate'] = stats['successes'] / stats['n']
                    aggregate['splits'][split] = stats
                    save(root / 'progress.json', aggregate)
                    print(json.dumps(dict(event='batch_complete', split=split, batch=batch_index, **stats)), flush=True)
                    del rows, gen_batch
            finally:
                env.close()
            assert len(episodes) == len(tasks) and len({r['episode_id'] for r in episodes}) == len(tasks)
            assert {r['game_id'] for r in episodes} == {t['game_id'] for t in tasks}
            by_family = collections.defaultdict(list)
            for episode in episodes:
                family = episode['task_family']
                by_family[family].append(episode['won'])
            aggregate['splits'][split]['by_family'] = {
                k: dict(n=len(v), successes=sum(v), success_rate=sum(v)/len(v)) for k, v in by_family.items()}
        aggregate.update(status='complete', elapsed_seconds=time.time() - start)
        save(root / 'results.json', aggregate)
        print(json.dumps(dict(event='evaluation_complete', **aggregate)), flush=True)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
