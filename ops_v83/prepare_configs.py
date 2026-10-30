"""Build full-size v8.3.1 phased configurations; this never launches training."""
import copy
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/v83_20261030'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True)
    parser.add_argument('--revision', type=int, required=True)
    args = parser.parse_args()
    release = json.loads((REPORT / f'release_{args.release}.json').read_text())
    model_contract = json.loads((ROOT / 'reports/model_contract.json').read_text())
    reference = json.loads((REPORT / 'a800_preflight.json').read_text())
    output = REPORT / f'prepared_configs_r{args.revision}'
    output.mkdir(exist_ok=True)
    summary = []
    for host in ['h800', 'a800']:
        preflight = json.loads((REPORT / f'{host}_preflight.json').read_text())
        root, vendor = Path(preflight['root']), Path(preflight['vendor'])
        source = root / 'v83_releases' / release['identity']
        for stage in (['tier1a', 'tier1b', 'tier2'] if host == 'h800' else ['tier1a', 'tier1b']):
            run_id = f'repro-v831-qwen25-{stage}-{host}-s0-r{args.revision}'
            run = root / 'runs' / run_id
            config = copy.deepcopy(preflight['original_config'])
            config['repro'] = {'enabled': False}
            config['repro_v83'] = dict(
                run_id=run_id, output_dir=str(run / 'v83'), config_hash='PENDING',
                model_contract_hash=canonical(model_contract), enable_tier1a=True,
                enable_tier1b=stage in {'tier1b', 'tier2'}, enable_tier2=stage == 'tier2',
                deployment_host=host, spec_version='8.3.1', tier2_pool_mode='mixed_only',
                credit=dict(gamma=.98, lambda_aux=.5, kappa_nav=.05, noop_cost=.05,
                            beta=.5, epsilon=1e-6))
            config['data']['train_files'] = str(root / 'v83_data/train.parquet')
            config['data']['val_files'] = str(root / 'v83_data/test.parquet')
            config['trainer'].update(experiment_name=run_id, project_name='repro-v831',
                default_local_dir=str(run / 'checkpoints'), resume_mode='disable',
                resume_from_path=None)
            # Ray adds a timestamped session and a Unix-domain socket suffix.
            # Use a short, run-unique directory to fit Linux's 107-byte limit.
            config['ray_init']['_temp_dir'] = '/tmp/rv831-' + hashlib.sha256(run_id.encode()).hexdigest()[:12]
            env = config['ray_init']['runtime_env']['env_vars']
            env['ALFWORLD_DATA'] = str(root / 'v83_data/alfworld')
            env['PYTHONPATH'] = ':'.join(map(str, [source, vendor,
                vendor / 'agent_system/environments/env_package/alfworld']))
            if host == 'a800':
                env['PATH'] = '/data_b/hhchang/wlx/repro/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
            expected = [(config['trainer']['total_epochs'], 150),
                        (config['data']['train_batch_size'], 16),
                        (config['env']['rollout']['n'], 8), (config['env']['max_steps'], 50),
                        (config['data']['max_prompt_length'], 2048),
                        (config['data']['max_response_length'], 512),
                        (config['trainer']['nnodes'], 1), (config['trainer']['n_gpus_per_node'], 2),
                        (config['actor_rollout_ref']['rollout']['tensor_model_parallel_size'], 2)]
            assert all(actual == wanted for actual, wanted in expected)
            assert config['actor_rollout_ref'] == preflight['original_config']['actor_rollout_ref']
            contract = dict(config=config, release_identity=release['identity'],
                release_files=release['files'], upstream=preflight['upstream_commit'],
                upstream_manifest_sha256=canonical(json.loads((ROOT / 'reports/official_source_manifest.json').read_text())),
                model_contract_sha256=canonical(model_contract),
                datasets_sha256=canonical(reference['datasets']),
                spec_sha256=digest(REPORT / 'implementation_spec.v831.original.md'),
                parquet_sha256={name: digest(REPORT / f'canonical_{name}.parquet')
                                for name in ['train', 'test']})
            config_hash = canonical(contract)
            # Deep copy the pre-hash contract so the self reference remains PENDING.
            contract = copy.deepcopy(contract)
            config['repro_v83']['config_hash'] = config_hash
            path = output / (run_id + '.yaml')
            if path.exists():
                raise FileExistsError('Prepared configurations are immutable; use a new revision')
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            card = dict(status='PREPARED_BLOCKED', prepared_at=datetime.now(timezone.utc).isoformat(),
                run_id=run_id, host=host, phase=stage, config_hash=config_hash,
                config_sha256=digest(path), release_identity=release['identity'],
                intended_gpu_indices=[0, 1] if host == 'h800' else [2, 3],
                initialization='Original Qwen2.5-1.5B-Instruct weights; no warm-start from another ablation',
                hypothesis='Tier1A separates aliased mixed-group anchors; Tier1B adds all-failed action credit; Tier2 tests history-conditioned potentials beyond fixed priors',
                alternative='Changes can reflect hardware stochasticity, task ordering, or integration effects rather than useful credit',
                primary_metric='Full-protocol ALFWorld success rate at fixed iteration 150; seen and unseen reported separately',
                secondary_metrics=['Per-family success', 'Invalid/no-op rates', 'AF group fraction',
                                   'Anchor sizes', 'Credit signs and scales', 'Wall time and GPU hours'],
                configuration=dict(iterations=150, task_groups=16, trajectories_per_group=8,
                                   horizon=50, prompt_tokens=2048, response_tokens=512, gpus=2),
                preceding_gate='Complete Step0 and UT1-8 plus native CPU integration checks' if stage == 'tier1a' else
                               'Complete previous stage on the SAME host/hardware/TP with verified step150 checkpoint and full evaluation',
                documented_errata='UT2 uses the verified F22-GiGPO step7 recovery; UT4/UT6 use the full original equation (+0.048/+0.04)',
                pending=['Latest complete native CPU checks and UT1-8',
                         'Fresh GPU UUID/process and frozen source/data/config validation'],
                tier2_status='Fully specified in v8.3.1; 96 cells, 4 preceding committed iterations, first-visit counts, alpha=5 and exact projection floors',
                automatic_launch=False, time_limit=None, gpu_hour_limit=None,
                runtime_optimizations=['One-time complete native task metadata preload with recorded startup cost',
                                       'Equivalent repeated anchor serialization reuse'],
                cpu_cost_policy='Observed CPU cost accepted by user v8.3.1; no hard 0.10-second gate, default garbage collection retained',
                stop_conditions=['Nonfinite optimizer update', 'Contract/hash mismatch',
                                 'Missing or duplicated trajectory identities', 'Resource or runtime failure'],
                interpretation='One seed is an engineering feasibility result; no statistical superiority claim',
                protocol_notes=['Both hosts use full identical data and canonical task enumeration',
                                'Canonical task order differs from historical baseline enumeration',
                                'Validation retains 128 samples in batches of 32 as in the prior runs',
                                '1A vs 1A+1B is compared within the same host; cross-host differences are not a causal ablation',
                                'H800 progresses 1A -> 1A+1B -> 1A+1B+2; A800 progresses 1A -> 1A+1B',
                                'Each stage restarts from the same original pretrained weights and retains the full protocol'])
            path.with_suffix('.contract.json').write_text(json.dumps(contract, indent=2) + '\n')
            path.with_suffix('.card.json').write_text(json.dumps(card, indent=2) + '\n')
            summary.append(dict(run_id=run_id, config_sha256=card['config_sha256'], status=card['status']))
    (output / 'index.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
