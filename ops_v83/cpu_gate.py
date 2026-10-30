"""Execute every CPU rule/integration test on the frozen deployment release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--compact', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release', required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    # Map remote absolute paths only to the already hash-verified shared data.
    compact = json.loads(args.compact.read_text())
    mappings = []
    for row in compact['rows']:
        original = row['gamefile']
        relative = Path(*Path(original).parts[-4:])
        target = args.data / 'json_2.1.1' / relative
        assert target.is_file(), str(target)
        row['gamefile'] = str(target)
        mappings.append(dict(original=original, mapped=str(target),
                             sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    fixture = args.output/'host_local_compact.json'
    fixture.write_text(json.dumps(compact)+'\n')
    (args.output/'fixture_mapping.json').write_text(json.dumps(mappings, indent=2)+'\n')
    os.environ.update(V83_NATIVE_CPU='1', V83_NATIVE_COLLECTOR_CPU='1',
                      V83_NATIVE_CONFIG=str(args.config), V83_COMPACT_TRACES=str(fixture),
                      V83_NATIVE_ROOT=str(root), ALFWORLD_DATA=str(args.data))
    sys.path.insert(0, str(root))
    loader = unittest.TestLoader()
    suite = loader.discover(str(root/'tests_v83'), pattern='test_*.py')
    for module in ['test_repro_v83_rules', 'test_repro_v831_table', 'test_repro_v831_runtime']:
        suite.addTests(loader.loadTestsFromName(module))
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    import torch
    assert not torch.cuda.is_initialized()
    summary = dict(status='passed' if result.wasSuccessful() and not result.skipped else 'failed',
                   release_identity=args.release, tests=result.testsRun,
                   failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                   cuda_initialized=torch.cuda.is_initialized(), seconds=time.monotonic()-started,
                   config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                   sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(root.rglob('*.py'))},
                   limitations=['CPU rules, original collector/batch path and checkpoint-state tests; no GPU training.'])
    (args.output/'suite_result.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='sources'}), flush=True)
    if summary['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__': main()
