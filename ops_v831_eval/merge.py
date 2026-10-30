"""Official two-rank FSDP merge plus full tensor check, with CUDA hidden."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    from .contract import digest, validate_frozen, write
except ImportError:
    from contract import digest, validate_frozen, write


def merge_commands(root, vendor, python):
    script = str(Path(vendor) / 'scripts/model_merger.py')
    common = ['--backend', 'fsdp', '--local_dir', str(Path(root) / 'checkpoint_actor')]
    return [[python, script, 'merge', *common, '--target_dir', str(Path(root) / 'hf_model')],
            [python, script, 'test', *common, '--test_hf_dir', str(Path(root) / 'hf_model')]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    card = validate_frozen(root)
    if (root / 'hf_model').exists() or (root / 'merge_verification.json').exists():
        raise FileExistsError('Use a fresh prepared evaluation for a new merge attempt')
    commands = merge_commands(root, card['source_root'], sys.executable)
    env = dict(os.environ, **card['entrypoint_environment'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    start = time.monotonic()
    for command, name in zip(commands, ['weight_merge.log', 'weight_verification.log']):
        with (root / name).open('x') as log:
            subprocess.run(command, cwd=card['source_root'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    validate_frozen(root)
    files = {str(p.relative_to(root / 'hf_model')): digest(p) for p in (root / 'hf_model').rglob('*') if p.is_file()}
    if 'model.safetensors' not in files:
        raise ValueError('Expected the single-file Qwen2.5-1.5B HF model')
    write(root / 'merge_verification.json', dict(status='passed', run_id=card['training_run_id'],
        freeze_sha256=digest(root / 'freeze.json'), commands=commands, cuda_visible_devices='',
        seconds=time.monotonic()-start, model_files=files,
        merge_log_sha256=digest(root / 'weight_merge.log'),
        full_tensor_verification_log_sha256=digest(root / 'weight_verification.log')))


if __name__ == '__main__':
    main()
