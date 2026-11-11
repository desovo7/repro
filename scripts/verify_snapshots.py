#!/usr/bin/env python3
"""Verify this source publication without importing training dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-submodule', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'snapshot_manifest.json').read_text())
    failures = []
    for relative, record in manifest['files'].items():
        path = root / relative
        if not path.is_file():
            failures.append(f'missing: {relative}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            failures.append(f'changed: {relative}')
    for name in ('webshop_r2', 'webshop_r2_aligned'):
        folder = root / 'experiments' / name
        config = json.loads((folder / 'frozen_run/config.json').read_text())
        for relative, expected in config['webshop_release']['files'].items():
            path = folder / 'webshop' / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                failures.append(f'run contract mismatch: {name}/{relative}')
    if args.with_submodule:
        submodule = root / manifest['upstream']['path']
        result = subprocess.run(['git', '-C', str(submodule), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True)
        if result.returncode or result.stdout.strip() != manifest['upstream']['commit']:
            failures.append('submodule missing or at a different commit')
        else:
            result = subprocess.run(['git', '-C', str(submodule), 'status', '--porcelain'],
                                    capture_output=True, text=True)
            if result.returncode or result.stdout.strip():
                failures.append('submodule working tree is not clean')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f"PASS: {len(manifest['files'])} published files and both WebShop runtime contracts")
    if args.with_submodule:
        print('PASS: pinned upstream submodule')


if __name__ == '__main__':
    main()
