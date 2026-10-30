"""Run the frozen standalone evaluator with its explicitly captured environment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    from .contract import begin_attempt, validate_ready, write
except ImportError:
    from contract import begin_attempt, validate_ready, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    card = validate_ready(root)
    env = dict(os.environ, **card['entrypoint_environment'])
    command = [sys.executable, '-u', str(root / 'evaluate.py'), '--root', str(root)]
    if args.preflight:
        env['CUDA_VISIBLE_DEVICES'] = ''
        command.append('--preflight')
    else:
        attempt = begin_attempt(root, card)
        env['V831_EVALUATION_ATTEMPT_ID'] = attempt['attempt_id']
    start = time.time()
    monotonic = time.monotonic()
    returncode = None
    try:
        result = subprocess.run(command, cwd=card['source_root'], env=env)
        returncode = result.returncode
    finally:
        seconds = time.monotonic()-monotonic
        write(root / ('preflight_cost.json' if args.preflight else 'evaluation_cost.json'), dict(
            run_id=card['training_run_id'], returncode=returncode, start=start,
            attempt_id=None if args.preflight else attempt['attempt_id'],
            elapsed_seconds=seconds, physical_gpus=0 if args.preflight else 2,
            gpu_hours=0 if args.preflight else 2*seconds/3600,
            accounting='Requested two-GPU reservation wall time including this attempt startup/failure'))
    raise SystemExit(returncode)


if __name__ == '__main__':
    main()
