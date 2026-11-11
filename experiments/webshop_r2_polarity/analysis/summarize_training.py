"""Recount the new branch's actual training and final native validation traces."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys

ROOT = Path('/data_b/qzluo/workspace/wlx/Repro-verl-agent/runs/gradus-webshop-h800-s0-polarity-from115-20261111')
sys.path.insert(0, str(ROOT))
from webshop.core import Config
from webshop.adapter import parse_trace

config = json.loads((ROOT/'config.json').read_text())
method = Config(**config['gradus_webshop'])
trace_file = ROOT/'gradus/train_traces.jsonl'
totals, rounds = Counter(), []
trace_hash = hashlib.sha256()
trajectory_ids, action_ids = set(), set()
task_sequence = {}


def source():
    with trace_file.open('rb') as f:
        for line in f:
            trace_hash.update(line)
            yield json.loads(line)


for iteration, group in itertools.groupby(source(), key=lambda t: t['iteration_id']):
    batch = list(group)
    assert iteration == 116 + len(rounds) and len(batch) == 128
    task_sequence[iteration] = [(t['task_uid'], t['instruction']) for t in batch]
    grouped = defaultdict(list)
    for raw in batch:
        assert raw['run_id'] == ROOT.name and raw['split'] == 'train'
        assert raw['trajectory_uid'] not in trajectory_ids
        trajectory_ids.add(raw['trajectory_uid'])
        grouped[raw['group_uid']].append(raw)
    assert len(grouped) == 16 and {len(v) for v in grouped.values()} == {8}
    af = {uid: not any(t['success'] for t in ts) for uid, ts in grouped.items()}
    count = Counter(trajectories=128, successes=sum(t['success'] for t in batch), af_groups=sum(af.values()))
    for raw in batch:
        parsed = parse_trace(raw, method)['trace']
        for i, step in enumerate(raw['steps']):
            assert step['action_uid'] not in action_ids
            action_ids.add(step['action_uid'])
            count['actions'] += 1
            count['format_invalid'] += not step['format_valid']
            count['semantic_noop'] += parsed.noops[i]
            count['return_navigation_all'] += parsed.return_navs[i]
            count['terminal_buy_all'] += parsed.terminal_buys[i]
            if af[raw['group_uid']]:
                count['af_actions'] += 1
                count['return_navigation_AF'] += parsed.return_navs[i]
                count['terminal_buy_AF'] += parsed.terminal_buys[i]
                count['format_invalid_AF'] += not step['format_valid']
    diag = json.loads((ROOT/f'gradus/diagnostics/iteration_{iteration}.json').read_text())
    metrics = diag['metrics']
    assert count['actions'] == metrics['actions']
    assert count['af_actions'] == metrics['af_actions']
    assert count['return_navigation_all'] == metrics['adapter_executable_return_navigation']
    assert count['format_invalid'] == metrics['adapter_format_invalid_actions']
    assert metrics['auxiliary_gated_nonzero_tokens'] == 0
    totals.update(count)
    rounds.append(dict(iteration=iteration, counts=dict(count),
        nonzero_centered_actions=metrics['nonzero_centered_actions'],
        gated_actor_rows=metrics['auxiliary_gated_actor_rows'],
        gated_response_tokens=metrics['auxiliary_gated_response_tokens'],
        gated_nonzero_tokens=metrics['auxiliary_gated_nonzero_tokens'],
        injection_rms=metrics['weighted_rms'],
        AF_clipped_fraction=metrics['af_clipped_token_fraction']))
assert len(rounds) == 35 and totals['trajectories'] == 4480
baseline = ROOT.parent/'gradus-webshop-h800-s0-r2aligned-20261109'
baseline_tasks = defaultdict(list)
with (baseline/'gradus/train_traces.jsonl').open() as f:
    for line in f:
        trace = json.loads(line)
        if trace['iteration_id'] >= 116:
            baseline_tasks[trace['iteration_id']].append((trace['task_uid'], trace['instruction']))
assert dict(baseline_tasks) == task_sequence, 'Training task sequence changed'
validation = []
for line in (ROOT/'gradus/validation_traces/iteration_150.jsonl').read_text().splitlines():
    raw = json.loads(line)
    last = raw['steps'][-1]
    timeout = not last['environment_done']
    if timeout:
        assert len(raw['steps']) == 15
        score = 0.0
    else:
        score = float(re.search(r'Your score \(min 0\.0, max 1\.0\)[^0-9]*([0-9.]+)', last['observation_after'])[1])
    assert 0 <= score <= 1 and (score == 1) == raw['success'] == last['won']
    validation.append(dict(uid=raw['trajectory_uid'], success=score == 1, score=score, timeout=timeout))
assert len(validation) == len({x['uid'] for x in validation}) == 256
out = dict(at=datetime.now(timezone.utc).isoformat(), status='PASS', run_id=ROOT.name,
    new_training_iterations=[116,150], inherited_complete_iteration=115,
    all4480_training_task_id_and_instruction_order_matches_original=True,
    trace_sha256=trace_hash.hexdigest(), counts=dict(totals), rounds=rounds,
    total_gated_rows=sum(x['gated_actor_rows'] for x in rounds),
    total_gated_tokens=sum(x['gated_response_tokens'] for x in rounds),
    total_gated_nonzero_tokens=sum(x['gated_nonzero_tokens'] for x in rounds),
    final_native_validation=dict(episodes=256, wins=sum(x['success'] for x in validation),
        success_rate=sum(x['success'] for x in validation)/256,
        task_score=sum(x['score'] for x in validation)/256,
        timeouts=sum(x['timeout'] for x in validation)),
    interpretation='Native validation is not the fixed panel; count-return targets must distinguish new116-150 from inherited100-115 and all trajectories from AF-only.')
(ROOT/'training_trace_recount.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rounds'}))
