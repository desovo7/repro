"""CPU-only contracts for portable v8.3.1 final-model evaluation."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

COUNTS = {'seen': 140, 'unseen': 134}
SPLITS = {'seen': 'valid_seen', 'unseen': 'valid_unseen'}
NATIVE_SPLITS = {'seen': 'eval_in_distribution', 'unseen': 'eval_out_of_distribution'}
FROZEN_PROTOCOL_SHA256 = 'b948997c823517103178d391d7e0d745bbc341d0565733042341f0c5b62960c3'
ACTOR_REQUIRED = {'config.json', 'generation_config.json', 'tokenizer.json',
                  'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt',
                  'model_world_size_2_rank_0.pt', 'model_world_size_2_rank_1.pt'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source_digest(path):
    path = Path(path)
    return hashlib.sha256(os.readlink(path).encode()).hexdigest() if path.is_symlink() else digest(path)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_protocol(protocol):
    expected = dict(episodes_per_task=1, seen_tasks=140, unseen_tasks=134, batch_size=32,
        model_seed=0, temperature=.4, top_p=1, top_k=-1, do_sample=True,
        max_steps=50, max_prompt_tokens=2048, max_response_tokens=512, history_length=2,
        retry_failed_tasks=False, training_updates=0)
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError('Frozen evaluation protocol mismatch: ' + key)
    if protocol.get('environment_seed') != '1000 + manifest_index, restarted per split':
        raise ValueError('Environment seed protocol changed')
    if protocol.get('seen_batch_sizes') != [32, 32, 32, 32, 12] or protocol.get('unseen_batch_sizes') != [32, 32, 32, 32, 6]:
        raise ValueError('Evaluation batch order changed')


def materialize_manifest(protocol_document, data_root):
    """Only map frozen relative paths; never resample/re-enumerate tasks."""
    validate_protocol(protocol_document['protocol'])
    data_root = Path(data_root).resolve()
    manifest = json.loads(json.dumps(protocol_document['task_manifest']))
    if set(manifest['splits']) != set(COUNTS):
        raise ValueError('Expected exactly seen and unseen splits')
    for split, n in COUNTS.items():
        tasks = manifest['splits'][split]['tasks']
        ids = [task['game_id'] for task in tasks]
        if len(tasks) != n or len(set(ids)) != n or ids != sorted(ids):
            raise ValueError('Missing, duplicate or unordered task IDs: ' + split)
        for index, task in enumerate(tasks):
            expected = Path('json_2.1.1') / SPLITS[split] / task['game_id'] / 'game.tw-pddl'
            relative = Path(task['relative_gamefile'])
            if relative != expected or relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Task path differs from its frozen split/identity')
            if task['manifest_index'] != index:
                raise ValueError('Task manifest index changed')
            task['gamefile'] = str(data_root / relative)
            if digest(task['gamefile']) != task['sha256']:
                raise ValueError('Task game hash changed: ' + task['game_id'])
    return manifest


def validate_checkpoint(checkpoint, config, base_model):
    checkpoint, base_model = Path(checkpoint), Path(base_model)
    marker = read(checkpoint / 'V83_COMPLETE.json')
    if checkpoint.name != 'global_step_150' or marker['iteration_id'] != 150:
        raise ValueError('Evaluation requires the fixed final step150 checkpoint')
    options = config['repro_v83']
    for key in ['run_id', 'config_hash', 'model_contract_hash']:
        if marker[key] != options[key]:
            raise ValueError('Checkpoint training identity mismatch: ' + key)
    state = read(checkpoint / 'v83_runtime.json')
    for name, key in [('v83_runtime.json', 'runtime_sha256'), ('v83_driver_rng.pt', 'rng_sha256')]:
        if digest(checkpoint / name) != marker[key]:
            raise ValueError('Checkpoint completion digest mismatch: ' + name)
    if any(marker.get(key) != value for key, value in state.items()):
        raise ValueError('Runtime state and completion marker disagree')
    if state.get('version') != '8.3.1' or state.get('enable_tier2') is not options['enable_tier2']:
        raise ValueError('Checkpoint stage/version mismatch')
    if state['enable_tier2'] and state['tier2_state']['last_committed_iteration'] != 150:
        raise ValueError('Incomplete Tier2 history')
    actor = checkpoint / 'actor'
    names = {p.name for p in actor.iterdir() if p.is_file()}
    shards = {name for name in names if name.startswith('model_world_size_')}
    if not ACTOR_REQUIRED.issubset(names) or shards != {f'model_world_size_2_rank_{i}.pt' for i in range(2)}:
        raise ValueError('Missing native two-rank model/tokenizer checkpoint files')
    metadata, base = read(actor / 'config.json'), read(base_model / 'config.json')
    for key in ['model_type', 'architectures', 'hidden_size', 'num_hidden_layers',
                'num_attention_heads', 'num_key_value_heads', 'vocab_size', 'tie_word_embeddings']:
        if metadata.get(key) != base.get(key):
            raise ValueError('Checkpoint model metadata mismatch: ' + key)
    if metadata.get('model_type') != 'qwen2':
        raise ValueError('Expected Qwen2.5 model architecture')
    # Retain every actor-root metadata/tokenizer/extra-state file. Only native
    # optimizer shards are omitted; a whitelist could silently drop new metadata.
    selected = sorted(name for name in names if not name.startswith('optim_world_size_'))
    if any((actor / name).stat().st_size == 0 for name in selected):
        raise ValueError('Empty checkpoint artifact')
    return marker, {name: digest(actor / name) for name in selected}


def gpu_inventory(indices, host, *, require_idle=False):
    if len(indices) != 2 or len(set(indices)) != 2:
        raise ValueError('Exactly two distinct physical GPU indices are required')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True)
    rows = {int(row[0].strip()): [value.strip() for value in row[1:]] for row in csv.reader(io.StringIO(raw))}
    selected = {index: rows[index] for index in indices}
    expected = {'h800': 'H800', 'a800': 'A800'}[host]
    if any(expected not in row[1] for row in selected.values()):
        raise ValueError('GPU model differs from the training host contract')
    if require_idle:
        running = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        uuids = {row[0] for row in selected.values()}
        if any(row and row[0].strip() in uuids for row in csv.reader(io.StringIO(running))):
            raise RuntimeError('Selected evaluation GPUs already have compute processes')
    return selected


def validate_frozen(root):
    root = Path(root).resolve()
    frozen = read(root / 'freeze.json')
    for name, expected in frozen['local_files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or digest(path) != expected:
            raise ValueError('Frozen evaluation artifact changed: ' + name)
    for name, expected in frozen['external_files'].items():
        if digest(name) != expected:
            raise ValueError('Frozen official source/model/data changed: ' + name)
    card = read(root / 'experiment_card.json')
    if card['resources']['tensor_parallel_size'] != 2 or len(card['gpu_uuids']) != 2:
        raise ValueError('Evaluation must use TP2 on two GPUs')
    manifest = materialize_manifest(read(root / 'evaluation_protocol.json'), card['dataset_root'])
    if manifest != read(root / 'task_manifest.json'):
        raise ValueError('Materialized task manifest differs from frozen protocol')
    return card


def validate_ready(root, *, check_environment=False, check_gpus=False):
    root = Path(root).resolve()
    card = validate_frozen(root)
    merged = read(root / 'merge_verification.json')
    if merged['status'] != 'passed' or merged['run_id'] != card['training_run_id']:
        raise ValueError('Merged model lacks a matching successful official CPU verification')
    if merged['freeze_sha256'] != digest(root / 'freeze.json'):
        raise ValueError('Merge verification belongs to another frozen evaluation')
    actual_files = {str(p.relative_to(root / 'hf_model')) for p in (root / 'hf_model').rglob('*') if p.is_file()}
    if actual_files != set(merged['model_files']):
        raise ValueError('Merged model file set changed')
    for name, expected in merged['model_files'].items():
        if digest(root / 'hf_model' / name) != expected:
            raise ValueError('Merged model changed: ' + name)
    if check_environment:
        if Path(os.environ.get('ALFWORLD_DATA', '')).resolve() != Path(card['dataset_root']).resolve():
            raise ValueError('ALFWORLD_DATA must point to the frozen v83_data root')
        if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
            raise ValueError('vLLM evaluation requires spawn')
        if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
            raise ValueError('Numeric GPU mapping requires CUDA_DEVICE_ORDER=PCI_BUS_ID')
    if check_gpus:
        current = gpu_inventory(card['resources']['physical_gpu_indices'], card['execution_host'], require_idle=True)
        if [row[0] for row in current.values()] != card['gpu_uuids']:
            raise ValueError('Physical GPU identities changed')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(map(str, card['resources']['physical_gpu_indices'])):
            raise ValueError('CUDA_VISIBLE_DEVICES must match the frozen two numeric GPU indices')
    return card


def begin_attempt(root, card):
    """Claim one evaluation attempt even if it fails before saving a batch."""
    root = Path(root)
    if any(root.glob('*_batch_*.jsonl')) or (root / 'results.json').exists():
        raise FileExistsError('Preserve previous evaluation evidence and use a fresh directory')
    attempt = dict(attempt_id=uuid.uuid4().hex, run_id=card['training_run_id'],
                   started_at=time.time(), status='started')
    with (root / 'evaluation_attempt.json').open('x') as stream:
        stream.write(json.dumps(attempt, indent=2) + '\n')
    return attempt


def publish_summary(root):
    root = Path(root).resolve()
    card = validate_ready(root)
    result, verification = read(root / 'results.json'), read(root / 'verification.json')
    cost = read(root / 'evaluation_cost.json')
    execution = read(root / 'execution.json')
    attempt = read(root / 'evaluation_attempt.json')
    if cost['run_id'] != card['training_run_id'] or cost['returncode'] != 0 or cost['physical_gpus'] != 2:
        raise ValueError('Missing successful same-run two-GPU evaluation execution evidence')
    if cost['attempt_id'] != execution['attempt_id'] or cost['attempt_id'] != attempt['attempt_id']:
        raise ValueError('Evaluation execution evidence belongs to different attempts')
    if (execution['run_id'] != card['training_run_id'] or execution['execution_host'] != card['execution_host']
            or execution['tensor_parallel_size'] != 2 or execution['gpu_inventory'] != card['gpu_inventory']
            or execution['physical_gpu_indices'] != card['resources']['physical_gpu_indices']):
        raise ValueError('Evaluation hardware differs from the frozen same-host TP2 contract')
    if result['status'] != 'complete' or result['checkpoint'] != 150 or verification['checkpoint'] != 150:
        raise ValueError('Final evaluation did not complete')
    for split, count in COUNTS.items():
        a, b = result['splits'][split], verification['splits'][split]
        if a['n'] != count or b['unique_tasks'] != count or b['missing_tasks'] or b['duplicate_tasks']:
            raise ValueError('Incomplete evaluation task coverage')
        if a['successes'] != b['successes']:
            raise ValueError('Independent recount disagrees')
    names = ['experiment_card.json', 'training_config.yaml', 'freeze.json', 'evaluation_protocol.json',
             'task_manifest.json', 'merge_verification.json', 'results.json', 'verification.json', 'evaluation_cost.json',
             'evaluation_attempt.json', 'execution.json',
             'checkpoint_proof/V83_COMPLETE.json', 'checkpoint_proof/v83_runtime.json', 'checkpoint_proof/v83_driver_rng.pt']
    names += [f'{split}_batch_{batch}.jsonl' for split in COUNTS for batch in range(5)]
    summary = dict(status='complete', run_id=card['training_run_id'], checkpoint_iteration=150,
        seen_games=140, unseen_games=134, config_hash=card['training_config_hash'],
        execution_host=card['execution_host'], phase=card['phase'], tensor_parallel_size=2,
        gpu_uuids=card['gpu_uuids'], splits=result['splits'], elapsed_seconds=result['elapsed_seconds'],
        gpu_hours=cost['gpu_hours'],
        hashes={name: digest(root / name) for name in names},
        model_files=read(root / 'merge_verification.json')['model_files'],
        limitations=['Single training seed and one sampled episode per task',
                     'Previously used diagnostic data; not a new blinded test set'])
    write(root / 'phase_evaluation.json', summary)
    return summary
