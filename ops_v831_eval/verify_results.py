"""Independent recount against the frozen exhaustive task manifest."""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

try:
    from .contract import validate_ready, publish_summary, read, write
except ImportError:
    from contract import validate_ready, publish_summary, read, write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    root = Path(root).resolve()
    validate_ready(root)
    result, manifest = read(root / 'results.json'), read(root / 'task_manifest.json')
    require(result['checkpoint'] == 150 and result['status'] == 'complete', 'Expected complete step150 results')
    require(set(result['splits']) == {'seen', 'unseen'}, 'Unexpected evaluation split')
    all_uids = set()
    report = dict(checkpoint=150, scope='Every frozen eligible task exactly once', splits={}, checks=[])
    expected_batches = {f'{split}_batch_{batch}.jsonl' for split in ['seen', 'unseen'] for batch in range(5)}
    require({p.name for p in root.glob('*_batch_*.jsonl')} == expected_batches, 'Expected exactly ten frozen batches')
    for split, expected_n in [('seen', 140), ('unseen', 134)]:
        tasks = manifest['splits'][split]['tasks']
        expected = {t['game_id']: t for t in tasks}
        episodes = []
        for batch in range(5):
            rows = [json.loads(line) for line in (root / f'{split}_batch_{batch}.jsonl').read_text().splitlines()]
            n = min(32, expected_n - 32 * batch)
            require(len(rows) == n and {r['env_index'] for r in rows} == set(range(n)), 'Incomplete or duplicated batch positions')
            for row in rows:
                task = expected[row['game_id']]
                require(row['split'] == split and row['batch_index'] == batch, 'Episode split/batch mismatch')
                require(row['manifest_index'] == task['manifest_index'] == batch * 32 + row['env_index'], 'Manifest ordering mismatch')
                require(row['gamefile'] == task['gamefile'] and row['task_family'] == task['task_family'], 'Task identity mismatch')
                require(row['episode_id'] and row['episode_id'] not in all_uids, 'Missing or duplicated episode identity')
                all_uids.add(row['episode_id'])
                trace = row['trajectory']
                require(len(trace) == row['steps'] and 1 <= len(trace) <= 50, 'Invalid episode horizon')
                require([s['step'] for s in trace] == list(range(1, len(trace) + 1)), 'Trajectory step sequence mismatch')
                require(type(row['won']) is bool and all(type(s['won']) is bool for s in trace), 'Won must be a boolean')
                require(row['won'] == trace[-1]['won'] == (row['episode_reward'] > 0), 'Final won/reward mismatch')
                require(row['won'] == any(s['reward'] > 0 for s in trace), 'Step reward/won mismatch')
                require(all(math.isfinite(s['reward']) for s in trace), 'Nonfinite trajectory reward')
                require(math.isclose(row['episode_reward'], sum(s['reward'] for s in trace), abs_tol=1e-6), 'Reward sum mismatch')
            episodes.extend(rows)
        require(len(episodes) == len({r['game_id'] for r in episodes}) == expected_n, 'Missing or duplicated tasks')
        require({r['game_id'] for r in episodes} == set(expected), 'Task set mismatch')
        wins = sum(r['won'] for r in episodes)
        stats = result['splits'][split]
        require(stats['n'] == stats['unique_games'] == stats['expected_tasks'] == expected_n, 'Summary count mismatch')
        require(stats['successes'] == wins, 'Success recount mismatch')
        require(math.isclose(stats['success_rate'], wins / expected_n, abs_tol=1e-12), 'Success rate mismatch')
        by_family = defaultdict(list)
        for row in episodes:
            by_family[row['task_family']].append(row['won'])
        require({k: len(v) for k, v in by_family.items()} == manifest['splits'][split]['by_family'], 'Family manifest count mismatch')
        require(set(stats['by_family']) == set(by_family), 'Summary family set mismatch')
        for family, outcomes in by_family.items():
            actual = stats['by_family'][family]
            require(actual['n'] == len(outcomes) and actual['successes'] == sum(outcomes), 'Family recount mismatch')
            require(math.isclose(actual['success_rate'], sum(outcomes) / len(outcomes), abs_tol=1e-12), 'Family success rate mismatch')
        report['splits'][split] = dict(n=expected_n, successes=wins, success_rate=wins / expected_n,
            missing_tasks=0, duplicate_tasks=0, unique_tasks=expected_n,
            by_family={k: dict(n=len(v), successes=sum(v), success_rate=sum(v) / len(v)) for k, v in by_family.items()})
    require(len(all_uids) == 274, 'Expected 274 distinct episode identities')
    validate_ready(root)
    report['checks'] = [
        'Frozen evaluator, native source, model, protocol and all task file hashes unchanged on this host',
        'Exactly 140 seen and 134 unseen distinct task IDs', 'No missing or duplicate tasks',
        'All 274 trajectory UUIDs distinct', 'Last active won agrees with positive episode reward',
        'Step rewards sum to episode rewards', 'Recount and every family count match saved summary']
    write(root / 'verification.json', report)
    publish_summary(root)
    return report


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1]), ensure_ascii=False))
