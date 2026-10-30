"""Freeze an evaluation directory from a completed local v8.3.1 run (no GPU use)."""
import argparse
import os
from pathlib import Path
import shutil
import tempfile

try:
    from .contract import (canonical, digest, source_digest, read, write, materialize_manifest,
                           validate_checkpoint, gpu_inventory, NATIVE_SPLITS, FROZEN_PROTOCOL_SHA256)
except ImportError:
    from contract import (canonical, digest, source_digest, read, write, materialize_manifest,
                          validate_checkpoint, gpu_inventory, NATIVE_SPLITS, FROZEN_PROTOCOL_SHA256)


def prepare(config_path, protocol_path, output, gpu_indices, ray_temp):
    from omegaconf import OmegaConf
    config_path, protocol_path, output = map(lambda p: Path(p).resolve(), (config_path, protocol_path, output))
    if digest(protocol_path) != FROZEN_PROTOCOL_SHA256:
        raise ValueError('Use the exact pre-training frozen evaluation_protocol.json')
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    card = read(config_path.with_suffix('.card.json'))
    contract = read(config_path.with_suffix('.contract.json'))
    if digest(config_path) != card['config_sha256'] or canonical(contract) != card['config_hash']:
        raise ValueError('Training config/card/contract mismatch')
    options = config['repro_v83']
    if options['config_hash'] != card['config_hash'] or options['run_id'] != card['run_id']:
        raise ValueError('Training run identity mismatch')
    frozen_config = read(config_path.with_suffix('.contract.json'))['config']
    resolved = dict(config, repro_v83=dict(options, config_hash='PENDING'))
    if resolved != frozen_config:
        raise ValueError('Resolved training config differs from its frozen contract')
    host = options['deployment_host']
    if host != card['host'] or options['spec_version'] != '8.3.1':
        raise ValueError('Training host or version mismatch')
    flags = tuple(options[k] for k in ['enable_tier1a', 'enable_tier1b', 'enable_tier2'])
    if flags != {'tier1a': (True, False, False), 'tier1b': (True, True, False), 'tier2': (True, True, True)}[card['phase']]:
        raise ValueError('Training phase mismatch')
    root = config_path.parent.parent
    if not output.is_relative_to(root / 'evaluations') or output.exists():
        raise ValueError('Use a new evaluation directory under the same host run root/evaluations')
    checkpoint = (Path(config['trainer']['default_local_dir']) / 'global_step_150').resolve()
    if checkpoint.parent != root / 'runs' / card['run_id'] / 'checkpoints':
        raise ValueError('Checkpoint does not belong to this host/run root')
    marker, actor_files = validate_checkpoint(checkpoint, config, config['actor_rollout_ref']['model']['path'])
    data_root = root / 'v83_data/alfworld'
    if Path(config['ray_init']['runtime_env']['env_vars']['ALFWORLD_DATA']).resolve() != data_root.resolve():
        raise ValueError('Training used a different dataset root')
    protocol = read(protocol_path)
    if protocol['checkpoints'] != [150] or protocol['inference']['tensor_parallel_size'] != 2:
        raise ValueError('Protocol must select fixed step150 and TP2')
    manifest = materialize_manifest(protocol, data_root)
    vendor = Path(config['ray_init']['runtime_env']['env_vars']['PYTHONPATH'].split(':')[1]).resolve()
    upstream = read(root / 'reports/official_source_manifest.json')
    if canonical(upstream) != contract['upstream_manifest_sha256']:
        raise ValueError('Official source manifest changed since training freeze')
    external = {}
    for name, record in upstream['files'].items():
        path = vendor / name
        if source_digest(path) != record['sha256']:
            raise ValueError('Official source changed: ' + name)
        external[str(path)] = digest(path)
    # Freeze the whole staged dataset/logic manifest, not just evaluator paths.
    data_manifest = read(root / 'v83_data/dataset_manifest.json')
    if canonical(data_manifest['datasets']) != contract['datasets_sha256']:
        raise ValueError('Staged dataset differs from training contract')
    for split, records in data_manifest['datasets'].items():
        base = data_root / ('logic' if split == 'logic' else 'json_2.1.1/' + split)
        for name, record in records.items():
            path = base / name
            if digest(path) != record['sha256']:
                raise ValueError('Staged data/logic changed: ' + str(path))
            external[str(path)] = digest(path)
    model_contract = read(root / 'reports/model_contract.json')
    if canonical(model_contract) != options['model_contract_hash']:
        raise ValueError('Base model contract differs from training')
    base_model = Path(config['actor_rollout_ref']['model']['path'])
    for name, record in model_contract['files'].items():
        path = base_model / name
        if digest(path) != record['sha256']:
            raise ValueError('Base model/tokenizer changed: ' + name)
        external[str(path)] = digest(path)
    gpus = gpu_inventory(gpu_indices, host)
    ray_temp = Path(ray_temp).absolute()
    if len(str(ray_temp).encode()) > 40 or ray_temp.exists():
        raise ValueError('Use an unused short Ray temporary path (at most 40 bytes)')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.v831-eval-', dir=output.parent))
    try:
        (temporary / 'checkpoint_actor').mkdir()
        (temporary / 'checkpoint_proof').mkdir()
        for name in actor_files:
            src, dst = checkpoint / 'actor' / name, temporary / 'checkpoint_actor' / name
            try:
                os.link(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)
            if digest(dst) != actor_files[name]:
                raise ValueError('Checkpoint changed during capture')
        for name in ['V83_COMPLETE.json', 'v83_runtime.json', 'v83_driver_rng.pt']:
            shutil.copy2(checkpoint / name, temporary / 'checkpoint_proof' / name)
        if read(temporary / 'checkpoint_proof/V83_COMPLETE.json') != marker:
            raise ValueError('Completion marker changed during capture')
        for name, key in [('v83_runtime.json', 'runtime_sha256'), ('v83_driver_rng.pt', 'rng_sha256')]:
            if digest(temporary / 'checkpoint_proof' / name) != marker[key]:
                raise ValueError('Checkpoint runtime/RNG changed during capture')
        shutil.copy2(config_path, temporary / 'training_config.yaml')
        if digest(temporary / 'training_config.yaml') != card['config_sha256']:
            raise ValueError('Training configuration changed during capture')
        write(temporary / 'training_card.json', card)
        write(temporary / 'training_contract.json', contract)
        write(temporary / 'official_source_manifest.json', upstream)
        write(temporary / 'model_contract.json', model_contract)
        write(temporary / 'dataset_manifest.json', data_manifest)
        shutil.copy2(protocol_path, temporary / 'evaluation_protocol.json')
        if digest(temporary / 'evaluation_protocol.json') != FROZEN_PROTOCOL_SHA256:
            raise ValueError('Evaluation protocol changed during capture')
        for script in ['evaluate.py', 'fixed_tasks.py', 'verify_results.py', 'contract.py', 'merge.py', 'run.py']:
            shutil.copy2(Path(__file__).with_name(script), temporary / script)
        write(temporary / 'task_manifest.json', manifest)
        uuids = [row[0] for row in gpus.values()]
        env = dict(ALFWORLD_DATA=str(data_root), PYTHONPATH=':'.join(map(str, [vendor,
            vendor / 'agent_system/environments/env_package/alfworld'])),
            CUDA_VISIBLE_DEVICES=','.join(map(str, gpu_indices)), CUDA_DEVICE_ORDER='PCI_BUS_ID',
            VLLM_WORKER_MULTIPROC_METHOD='spawn',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        evaluation_card = dict(training_run_id=card['run_id'], training_config_hash=card['config_hash'],
            phase=card['phase'], execution_host=host, checkpoint=150, source_checkpoint=str(checkpoint),
            checkpoint_marker_sha256=digest(temporary / 'checkpoint_proof/V83_COMPLETE.json'),
            source_root=str(vendor), source_commit=upstream['commit'], tokenizer_path=str(base_model),
            dataset_root=str(data_root), protocol=protocol['protocol'], splits=NATIVE_SPLITS,
            resources=dict(physical_gpu_indices=gpu_indices, tensor_parallel_size=2, cpu_env_workers=32),
            gpu_uuids=uuids, gpu_inventory={str(k): v for k, v in gpus.items()},
            gpu_memory_utilization=.2, ray_temp_dir=str(ray_temp), entrypoint_environment=env,
            no_training_hooks=True,
            data_visibility=protocol['data_visibility'])
        write(temporary / 'experiment_card.json', evaluation_card)
        write(temporary / 'checkpoint_capture.json', dict(source=str(checkpoint), iteration_id=150,
            run_id=card['run_id'], actor_files=actor_files, optimizer_copied=False))
        local_files = {str(p.relative_to(temporary)): digest(p) for p in temporary.rglob('*') if p.is_file()}
        write(temporary / 'freeze.json', dict(schema='v831-fixed-eval-v1', local_files=local_files, external_files=external))
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu-indices', type=int, nargs=2, required=True)
    parser.add_argument('--ray-temp', required=True)
    args = parser.parse_args()
    print(prepare(args.config, args.protocol, args.output, args.gpu_indices, args.ray_temp))


if __name__ == '__main__':
    main()
