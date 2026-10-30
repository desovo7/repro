"""Verify a frozen v8.3 run and optionally submit it as a persistent user service."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def main():
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    config_path = args.config.resolve()
    card = json.loads(config_path.with_suffix('.card.json').read_text())
    contract = json.loads(config_path.with_suffix('.contract.json').read_text())
    cfg = OmegaConf.load(config_path)
    root = config_path.parent.parent
    release = root / 'v83_releases' / card['release_identity']
    run = root / 'runs' / card['run_id']
    assert sha(config_path) == card['config_sha256'], 'Config bytes changed'
    assert canonical(contract) == card['config_hash'], 'Contract changed'
    assert cfg.repro_v83.config_hash == card['config_hash']
    assert card['run_id'] == cfg.repro_v83.run_id
    assert card['release_identity'] == contract['release_identity']
    frozen = OmegaConf.to_container(cfg, resolve=True)
    frozen['repro_v83']['config_hash'] = 'PENDING'
    assert frozen == contract['config'], 'Resolved config differs from contract'
    for name, digest in contract['release_files'].items():
        assert sha(release / name) == digest, 'Release changed: ' + name
    for name, digest in contract['parquet_sha256'].items():
        assert sha(root / 'v83_data' / (name + '.parquet')) == digest
    datasets = json.loads((root / 'v83_data/dataset_manifest.json').read_text())
    assert canonical(datasets['datasets']) == contract['datasets_sha256']
    data = Path(datasets['data_root'])
    assert data.resolve() == Path(cfg.ray_init.runtime_env.env_vars.ALFWORLD_DATA).resolve(), 'Checked dataset is not the configured dataset'
    for split, files in datasets['datasets'].items():
        base = data / ('logic' if split == 'logic' else 'json_2.1.1/' + split)
        paths = [p for p in base.rglob('*') if p.is_file()] if split == 'logic' else [
            p for name in ['game.tw-pddl', 'traj_data.json'] for p in base.rglob(name) if p.is_file()]
        assert {str(p.relative_to(base)) for p in paths} == set(files), 'Task/logic file set changed'
        for name, record in files.items():
            assert sha(base / name) == record['sha256'], 'Game/metadata/logic changed: ' + name
    vendor = Path(cfg.ray_init.runtime_env.env_vars.PYTHONPATH.split(':')[1])
    manifest = json.loads((root / 'reports/official_source_manifest.json').read_text())
    assert canonical(manifest) == contract['upstream_manifest_sha256'], 'Official source reference manifest changed'
    assert manifest['commit'] == contract['upstream']
    for name, record in manifest['files'].items():
        path = vendor / name
        actual = hashlib.sha256(os.readlink(path).encode()).hexdigest() if path.is_symlink() else sha(path)
        assert actual == record['sha256'], 'Official source changed: ' + name
    model_contract = json.loads((root / 'reports/model_contract.json').read_text())
    assert canonical(model_contract) == contract['model_contract_sha256']
    for name, record in model_contract['files'].items():
        path = Path(cfg.actor_rollout_ref.model.path) / name
        assert path.stat().st_size == record['size'] and sha(path) == record['sha256']
    assert not cfg.repro.enabled
    assert cfg.repro_v83.spec_version == '8.3.1'
    assert cfg.repro_v83.deployment_host == card['host']
    expected_flags = {'tier1a': (True, False, False), 'tier1b': (True, True, False),
                      'tier2': (True, True, True)}[card['phase']]
    assert tuple(cfg.repro_v83[k] for k in ['enable_tier1a', 'enable_tier1b', 'enable_tier2']) == expected_flags
    assert cfg.trainer.nnodes == 1 and cfg.trainer.n_gpus_per_node == 2
    assert cfg.trainer.total_epochs == 150 and cfg.trainer.total_training_steps is None
    assert cfg.data.train_batch_size == 16 and cfg.env.rollout.n == 8 and cfg.env.max_steps == 50
    assert cfg.data.max_prompt_length == 2048 and cfg.data.max_response_length == 512
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
    assert cfg.trainer.resume_mode == 'disable', 'Resume requires its own verified launch contract'
    socket_suffix = '/session_2000-01-01_00-00-00_000000_9999999/sockets/plasma_store'
    assert len((str(cfg.ray_init._temp_dir) + socket_suffix).encode()) <= 107, 'Ray socket path is too long'
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
    gpu_map = {int(row[0].strip()): [x.strip() for x in row[1:]] for row in csv.reader(io.StringIO(inventory))}
    devices = card['intended_gpu_indices']
    selected = {gpu_map[i][0] for i in devices}
    applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                           '--format=csv,noheader,nounits'], text=True)
    occupied = [row for row in csv.reader(io.StringIO(applications)) if row and row[0].strip() in selected]
    gate_path = config_path.with_suffix('.gate.json')
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    reasons = []
    if occupied:
        reasons.append('Selected GPUs currently have compute processes')
    if gate.get('status') != 'PASSED' or gate.get('config_sha256') != card['config_sha256']:
        reasons.append('Stage evidence gate has not passed for this configuration')
    elif not all(gate.get(k) is True for k in ['spec_errata_resolved', 'step0_complete', 'real_trace_rules_passed',
                                      'native_cpu_checks_passed', 'tier2_rules_passed', 'cpu_cost_accepted', 'preceding_phase_passed']):
        reasons.append('One or more required specification/phase checks are missing')
    else:
        # Bind a stage decision to its reviewable artifacts, rather than just
        # accepting a collection of unversioned status flags.
        required = {'specification', 'spec_errata', 'step0_report', 'rule_audit', 'native_cpu_checks', 'native_flow', 'cpu_cost_report'}
        evidence = gate.get('evidence_files', {})
        assert required.issubset(evidence), 'Gate evidence is incomplete'
        for item in evidence.values():
            assert sha(Path(item['path'])) == item['sha256'], 'Gate evidence changed'
        assert evidence['specification']['sha256'] == contract['spec_sha256']
        checks = json.loads(Path(evidence['native_cpu_checks']['path']).read_text())
        assert checks['status'] == 'passed' and checks['failures'] == 0 and checks['errors'] == 0
        assert checks['skipped'] == 0 and checks['release_identity'] == card['release_identity']
        assert checks['config_sha256'] == card['config_sha256'] and checks['tests'] > 0
        expected_python = {name: digest for name, digest in contract['release_files'].items() if name.endswith('.py')}
        assert checks['sources'] == expected_python, 'CPU checks did not cover the frozen source release'
        flow = json.loads(Path(evidence['native_flow']['path']).read_text())
        assert flow['collector']['status'] == flow['ray_entrypoint']['status'] == 'passed'
        assert flow['config_sha256'] == card['config_sha256'] and flow['source_sha256'] == expected_python
        if card['phase'] in {'tier1b', 'tier2'}:
            predecessor = gate['predecessor']
            records = {}
            for name in ['config', 'checkpoint_complete', 'evaluation']:
                item = predecessor[name]
                path = Path(item['path'])
                assert sha(path) == item['sha256'], 'Preceding stage evidence changed'
                records[name] = OmegaConf.to_container(OmegaConf.load(path), resolve=True) if name == 'config' else json.loads(path.read_text())
            prior, checkpoint, evaluation = (records[k] for k in ['config', 'checkpoint_complete', 'evaluation'])
            assert prior['repro_v83']['enable_tier1a'] is True
            assert prior['repro_v83']['enable_tier1b'] is (card['phase'] == 'tier2')
            assert prior['repro_v83']['enable_tier2'] is False
            assert prior['repro_v83']['deployment_host'] == card['host'], 'Controlled ablation requires the SAME host'
            assert prior['actor_rollout_ref'] == OmegaConf.to_container(cfg.actor_rollout_ref, resolve=True)
            assert prior['data'] == OmegaConf.to_container(cfg.data, resolve=True)
            assert prior['env'] == OmegaConf.to_container(cfg.env, resolve=True)
            assert prior['algorithm'] == OmegaConf.to_container(cfg.algorithm, resolve=True)
            assert prior['trainer']['total_epochs'] == 150 and checkpoint['iteration_id'] == 150
            assert checkpoint['config_hash'] == prior['repro_v83']['config_hash']
            assert checkpoint['run_id'] == prior['repro_v83']['run_id'] == predecessor['run_id']
            assert evaluation['status'] == 'complete' and evaluation['run_id'] == predecessor['run_id']
            assert evaluation['checkpoint_iteration'] == 150
            assert evaluation['seen_games'] == 140 and evaluation['unseen_games'] == 134
    if run.exists() and any(run.iterdir()):
        reasons.append('Run directory already contains evidence; use recovery workflow')
    result = dict(host=card['host'], run_id=card['run_id'], checked_at=datetime.now(timezone.utc).isoformat(),
                  verified_source_data_model_config=True, gpu_inventory={str(i): gpu_map[i] for i in devices},
                  occupied=occupied, blockers=reasons, launched=False)
    if not args.execute:
        print(json.dumps(result, indent=2))
        return
    if reasons:
        raise RuntimeError('Cannot launch: ' + '; '.join(reasons))
    run.mkdir(parents=True, exist_ok=True)
    Path(cfg.ray_init._temp_dir).mkdir(parents=True, exist_ok=False)
    env = dict(cfg.ray_init.runtime_env.env_vars)
    # Pinned vLLM parses CUDA_VISIBLE_DEVICES entries as physical integer IDs.
    # Keep the verified index->UUID inventory in launch.json for provenance.
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, devices)), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               RAY_ADDRESS='local', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    python = str(root / '.venv/bin/python') if card['host'] == 'h800' else '/data_b/hhchang/wlx/repro/.venv/bin/python'
    argv = [python, '-m', 'repro_v83.entrypoint', str(config_path)]
    unit = 'codex-' + card['run_id'] + '.service'
    user = os.environ['USER']
    linger = subprocess.check_output(['loginctl', 'show-user', user, '--property=Linger', '--value'], text=True).strip()
    stop = 'echo service_result=$SERVICE_RESULT exit_code=$EXIT_CODE exit_status=$EXIT_STATUS > ' + shlex.quote(str(run / 'service_result.properties'))
    stop += '; /usr/bin/date -u --iso-8601=seconds >> ' + shlex.quote(str(run / 'service_result.properties'))
    if linger == 'no':
        stop += '; /usr/bin/loginctl --no-ask-password disable-linger ' + shlex.quote(user)
    properties = ['Type=exec', 'KillMode=control-group', 'TimeoutStopSec=30s', 'RuntimeMaxSec=infinity',
                  'Restart=no', 'LimitCORE=0', 'WorkingDirectory=' + str(vendor),
                  'StandardOutput=append:' + str(run / 'train.log'),
                  'StandardError=append:' + str(run / 'train.log'), 'ExecStopPost=/bin/sh -c ' + shlex.quote(stop)]
    result.update(argv=argv, env=env, unit=unit, properties=properties, original_linger=linger,
                  config_sha256=card['config_sha256'], gate_sha256=sha(gate_path))
    (run / 'spec.snapshot.json').write_text(json.dumps(result, indent=2) + '\n')
    if linger == 'no':
        subprocess.run(['loginctl', '--no-ask-password', 'enable-linger', user], check=True)
    command = ['systemd-run', '--user', '--unit=' + unit]
    command += ['--property=' + x for x in properties]
    command += ['--setenv=' + k + '=' + v for k, v in env.items()]
    submission = subprocess.run(command + ['--'] + argv, text=True, capture_output=True)
    (run / 'submission.json').write_text(json.dumps(dict(returncode=submission.returncode,
        stdout=submission.stdout, stderr=submission.stderr), indent=2) + '\n')
    if submission.returncode:
        if linger == 'no':
            subprocess.run(['loginctl', '--no-ask-password', 'disable-linger', user], check=True)
        raise RuntimeError(submission.stderr)
    result['launched'] = True
    result['service_properties'] = subprocess.check_output(['systemctl', '--user', 'show', unit,
        '-p', 'MainPID', '-p', 'ActiveState', '-p', 'SubState', '-p', 'ControlGroup'], text=True)
    (run / 'launch.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
