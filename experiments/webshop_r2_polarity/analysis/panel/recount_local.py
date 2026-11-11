"""Independently recount copied raw terminal observations and paired intervals."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import numpy as np
from native_projection import webshop_projection

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT/'task_manifest.json').read_text())['tasks']
task_ids = [t['task_uid'] for t in manifest]
tasks = {t['task_uid']:t for t in manifest}
comparison = json.loads((ROOT/'comparison.json').read_text())
summaries, arrays = {}, {}
NAV = {'buy now','back to search','< prev','next >','description','features','reviews','attributes','search'}

for tag in ['r2_150','aligned_150','gigpo_150']:
    folder = ROOT/'jobs'/tag
    verification = json.loads((folder/'verification.json').read_text())
    records = {}
    action_counts = dict(format_invalid_actions=0, unexecutable_actions=0, repeated_option_actions=0,
                         option_loop=0, failed_buy=0, steps=0)
    for path in sorted(folder.glob('seed_*_batch_*.jsonl')):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == verification['source_hashes'][path.name]
        for line in raw.decode().splitlines():
            rec = json.loads(line)
            uid, seed = rec['task_uid'], rec['replicate_seed']
            assert rec['tag'] == ('r2_150' if tag == 'aligned_150' else tag) and rec['initial_signature'] == tasks[uid]['initial_signature']
            assert (uid, seed) not in records
            last = rec['steps'][-1]
            timeout = not last['environment_done']
            if timeout:
                assert len(rec['steps']) == 15
                score = 0.0
            else:
                score = float(re.search(r'Your score \(min 0\.0, max 1\.0\)[^0-9]*([0-9.]+)', last['observation_after'])[1])
            won = score == 1.0
            assert 0 <= score <= 1 and abs(score-rec['task_score']) < 1e-12
            assert won == rec['success'] and rec['episode_reward'] == 10*won
            repeats = 0
            for j, step in enumerate(rec['steps']):
                parsed, valid = webshop_projection([step['generated_text']])
                assert parsed[0] == step['action'] and bool(valid[0]) == step['format_valid']
                action = step['action'].lower()
                clickables = {str(v).lower() for v in step['available_before']['clickables']}
                option = (action.startswith('click[') and action[6:-1] in clickables
                          and action[6:-1] not in NAV and not re.fullmatch(r'click\[b[a-z0-9]{9}\]',action))
                repeats += bool(j and option and action == rec['steps'][j-1]['action'].lower())
                executable = ((action.startswith('search[') and step['available_before']['has_search_bar'])
                              or (action.startswith('click[') and action[6:-1] in clickables))
                action_counts['format_invalid_actions'] += not valid[0]
                action_counts['unexecutable_actions'] += not executable
                action_counts['steps'] += 1
            action_counts['repeated_option_actions'] += repeats
            action_counts['option_loop'] += repeats >= 3
            action_counts['failed_buy'] += not won and last['action'].lower() == 'click[buy now]'
            records[(uid,seed)] = dict(success=float(won), score=score, timeout=float(timeout), steps=len(rec['steps']))
    assert set(records) == {(uid,seed) for uid in task_ids for seed in [0,1,2]}
    means = {k:statistics.mean(r[k] for r in records.values()) for k in ['success','score','timeout','steps']}
    for k,v in means.items():
        assert abs(v-verification['means'][k]) < 1e-12
    for k,v in action_counts.items():
        assert v == verification['counts'][k], (tag,k,v,verification['counts'][k])
    summaries[tag] = dict(episodes=len(records), wins=sum(r['success'] for r in records.values()),
        timeouts=sum(r['timeout'] for r in records.values()), means=means, action_counts=action_counts,
        raw_terminal_recount='passed', native_projection_and_action_recount='passed')
    arrays[tag] = {k:np.array([[records[(uid,seed)][k] for seed in [0,1,2]] for uid in task_ids]).mean(axis=1)
                   for k in means}

indices = np.random.default_rng(20261109).integers(0,512,size=(10000,512))
contrasts = []
for candidate,reference in [(x['candidate'],x['reference']) for x in comparison['paired_contrasts']]:
    expected = next(x for x in comparison['paired_contrasts'] if x['candidate']==candidate and x['reference']==reference)
    result = dict(candidate=candidate,reference=reference,metrics={})
    for metric in ['success','score','timeout','steps']:
        delta = arrays[candidate][metric]-arrays[reference][metric]
        difference = float(delta.mean())
        ci = np.quantile(delta[indices].mean(axis=1),[.025,.975])
        assert abs(difference-expected['metrics'][metric]['difference']) < 1e-12
        assert np.allclose(ci,expected['metrics'][metric]['ci95'],atol=1e-12,rtol=0)
        result['metrics'][metric] = dict(difference=difference,ci95=ci.tolist())
    contrasts.append(result)

out = dict(verified_at=datetime.now(timezone.utc).isoformat(), valid_episodes=4608,
    raw_recount=summaries, paired_contrasts=contrasts,
    method='Parse terminal observations from copied raw JSONL; average three repeats per task; paired task-cluster bootstrap with10000 resamples, seed20261109. Matches remote independent verifier.')
(ROOT/'local_raw_recount.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
(ROOT/'local_paired_recount.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False))
