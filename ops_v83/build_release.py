"""Freeze a reviewed source snapshot; never modify an existing release."""
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    root = Path(__file__).resolve().parents[1]
    report = root/'reports/v83_20261030'
    files = []
    for package in ['ops_v83', 'repro_core', 'repro_external', 'repro_integration', 'repro_v83', 'tests_v83']:
        files.extend(root.joinpath(package).rglob('*.py'))
    for name in ['run_step0_audit.py', 'test_repro_v83_rules.py',
                 'test_repro_v831_table.py', 'test_repro_v831_runtime.py']:
        files.append(root/name)
    for name in ['a800_preflight.json', 'benchmark_first_iteration_compact.json',
                 'native_flow_probe.py', 'replayed_traces.jsonl', 'implementation_spec.v831.original.md',
                 'implementation_errata.v831.json', 'v831_decisions.json']:
        files.append(report/name)
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:12]
    manifest = dict(identity=identity, version='8.3.1', files=hashes)
    destination = report/f'release_{identity}.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == manifest
    else:
        destination.write_text(json.dumps(manifest, indent=2)+'\n')
        with tarfile.open(report/f'release_{identity}.tar.gz', 'w:gz') as archive:
            for name in hashes: archive.add(root/name, arcname=name, recursive=False)
    print(json.dumps(dict(identity=identity, files=len(hashes), manifest=str(destination))))


if __name__ == '__main__': main()
