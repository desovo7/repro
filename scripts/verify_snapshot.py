"""Verify this publication without loading models or allocating GPUs."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def main():
    publication = json.loads((ROOT / 'snapshot_manifest.json').read_text())
    errors = []
    for name, expected in publication['published_files'].items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            errors.append('Published file differs: ' + name)
    release = json.loads((ROOT / 'reports/v83_20261030/release_0607ff62e225.json').read_text())
    if canonical(release['files'])[:12] != release['identity']:
        errors.append('Release identity differs')
    for name, expected in release['files'].items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != expected:
            errors.append('Frozen release differs: ' + name)
    contract = json.loads((ROOT / 'configs/default.contract.json').read_text())
    card = json.loads((ROOT / 'configs/default.card.json').read_text())
    if canonical(contract) != card['config_hash']:
        errors.append('Default configuration contract differs')
    if sha(ROOT / 'configs/default.yaml') != card['config_sha256']:
        errors.append('Default configuration bytes differ')
    config = contract['config']
    flags = {key: config['repro_v83'][key] for key in publication['enabled']}
    if flags != publication['enabled']:
        errors.append('Default algorithm switches differ')
    full_size = (config['trainer']['total_epochs'], config['data']['train_batch_size'],
                 config['env']['rollout']['n'], config['env']['max_steps'],
                 config['data']['max_prompt_length'], config['data']['max_response_length'],
                 config['trainer']['n_gpus_per_node'],
                 config['actor_rollout_ref']['rollout']['tensor_model_parallel_size'])
    if full_size != (150, 16, 8, 50, 2048, 512, 2, 2):
        errors.append('Full experiment configuration differs')
    entry = subprocess.check_output(['git', 'ls-files', '--stage', '--', 'vendor/verl_agent'],
                                    cwd=ROOT, text=True).split()
    if len(entry) < 2 or entry[:2] != ['160000', publication['upstream_commit']]:
        errors.append('Pinned upstream gitlink differs')
    print(json.dumps(dict(status='failed' if errors else 'passed', variant=publication['variant'],
                          release=release['identity'], frozen_files=len(release['files']),
                          published_files=len(publication['published_files']),
                          enabled=flags, errors=errors), indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
