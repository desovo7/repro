"""Create and verify a separate full ALFWorld dataset without changing old data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', choices=['h800', 'a800'], required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    h800 = args.host == 'h800'
    root = Path('/data_b/qzluo/workspace/wlx/Repro-verl-agent' if h800 else
                '/data_b/hhchang/wlx/verl-agent-runs')
    original = Path('/home/qzluo/.cache/alfworld' if h800 else
                    '/data_b/hhchang/wlx/repro/data/alfworld')
    matched_eval = root / 'evaluations/repro-step150-full-seen-unseen-20261030/alfworld_eval_data' if h800 else original
    target = root / 'v83_data/alfworld'
    marker = root / 'v83_data/dataset_manifest.json'
    if target.exists() and not marker.exists():
        raise FileExistsError('Incomplete staging directory requires investigation before reuse')
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir()
        for split in ['train', 'valid_seen', 'valid_unseen']:
            source = (original if split == 'train' else matched_eval) / 'json_2.1.1' / split
            shutil.copytree(source, target / 'json_2.1.1' / split)
        shutil.copytree(original / 'logic', target / 'logic')
    actual = {}
    for split in ['train', 'valid_seen', 'valid_unseen', 'logic']:
        base = target / ('logic' if split == 'logic' else 'json_2.1.1/' + split)
        paths = sorted(base.rglob('*')) if split == 'logic' else sorted(
            p for name in ['game.tw-pddl', 'traj_data.json'] for p in base.rglob(name))
        actual[split] = {str(p.relative_to(base)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                         for p in paths if p.is_file()}
        if actual[split] != reference['datasets'][split]:
            raise ValueError('Staged dataset differs from the cross-host reference: ' + split)
    result = dict(status='verified', captured_at=datetime.now(timezone.utc).isoformat(),
                  host=args.host, data_root=str(target), datasets=actual,
                  source_train=str(original), source_eval=str(matched_eval),
                  scope='Full native train/seen/unseen directories; all game, task metadata and logic files verified',
                  reference_sha256=sha(args.reference))
    marker.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'datasets'}))
    print(json.dumps({k: len(v) for k, v in actual.items()}))


if __name__ == '__main__':
    main()
