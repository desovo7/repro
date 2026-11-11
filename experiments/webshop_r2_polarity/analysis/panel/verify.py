"""Independent episode recount and paired task-cluster uncertainty."""
import argparse
from collections import Counter
import importlib.util
import json
import re
from pathlib import Path
import statistics
from common import ROOT, SOURCE, digest, now, read, request_seed, verify_frozen, write

NAV={'buy now','back to search','< prev','next >','description','features','reviews','attributes','search'}


def verify(tag):
    card=verify_frozen()
    module_spec=importlib.util.spec_from_file_location('frozen_webshop_projection',SOURCE/'agent_system/environments/env_package/webshop/projection.py')
    projection=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(projection)
    manifest=read(ROOT/'task_manifest.json')['tasks']
    tasks={t['task_uid']:t for t in manifest}
    job=ROOT/'jobs'/tag
    complete=read(job/'execution_complete.json')
    expected_hash={(b['replicate'],b['batch']):b['sha256'] for b in complete['batches']}
    pairs={};uuids=set();features=[];source_hashes={}
    for seed in card['replicate_seeds']:
        for batch in range(4):
            path=job/f'seed_{seed}_batch_{batch}.jsonl'
            sha=digest(path)
            assert sha==expected_hash[(seed,batch)]
            source_hashes[path.name]=sha
            records=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            assert len(records)==128
            for index,rec in enumerate(records):
                task=tasks[rec['task_uid']]
                assert task==manifest[batch*128+index]
                assert rec['tag']==card.get('expected_record_tags', {}).get(tag, tag) and rec['replicate_seed']==seed and rec['batch']==batch
                for key in ['panel_index','instruction','initial_signature','goal_hash','worker_seed','goal_index','price_hash']:
                    assert rec[key]==task[key],key
                key=(rec['task_uid'],seed)
                assert key not in pairs and rec['episode_uid'] not in uuids
                uuids.add(rec['episode_uid'])
                steps=rec['steps'];assert 1<=len(steps)<=15
                assert [s['turn'] for s in steps]==list(range(len(steps)))
                assert not any(s['environment_done'] for s in steps[:-1])
                timeout=not steps[-1]['environment_done']
                if timeout:
                    assert len(steps)==15
                    score=0.0
                else:
                    match=re.search(r'Your score \(min 0\.0, max 1\.0\)[^0-9]*([0-9.]+)',steps[-1]['observation_after'])
                    assert match,('Missing terminal score',key)
                    score=float(match[1])
                assert 0<=score<=1 and abs(score-rec['task_score'])<1e-12
                won=(score==1.0)
                assert won==rec['success']==steps[-1]['won']
                assert rec['episode_reward']==10*won
                repeated_options=0;invalid_format=0;unexecutable=0
                classes=[]
                for j,s in enumerate(steps):
                    assert s['sampling_seed']==request_seed(task['task_uid'],seed,j)
                    parsed,valid=projection.webshop_projection([s['generated_text']])
                    assert parsed[0]==s['action'] and bool(valid[0])==s['format_valid'],('Recorded action differs from native projection',key,j)
                    action=s['action'].lower()
                    clickable={str(x).lower() for x in s['available_before']['clickables']}
                    option=action.startswith('click[') and action[6:-1] in clickable and action[6:-1] not in NAV and not re.fullmatch(r'click\[b[a-z0-9]{9}\]',action)
                    classes.append('option' if option else 'other')
                    if j and option and action==steps[j-1]['action'].lower():
                        repeated_options+=1
                    executable=(action.startswith('search[') and s['available_before']['has_search_bar']) or (action.startswith('click[') and action[6:-1] in clickable)
                    unexecutable+=not executable
                    invalid_format+=not s['format_valid']
                row=dict(task_uid=task['task_uid'],panel_index=task['panel_index'],replicate_seed=seed,
                    success=float(won),score=score,timeout=float(timeout),steps=len(steps),
                    failed_buy=float(not won and steps[-1]['action'].lower()=='click[buy now]'),
                    option_loop=float(repeated_options>=3),repeated_option_actions=repeated_options,
                    format_invalid_actions=invalid_format,unexecutable_actions=unexecutable)
                features.append(row);pairs[key]=row
    assert set(pairs)=={(t,seed) for t in tasks for seed in card['replicate_seeds']}
    assert len(features)==1536
    means={k:statistics.mean(x[k] for x in features) for k in ['success','score','timeout','steps','failed_buy','option_loop']}
    counts={k:sum(x[k] for x in features) for k in ['success','timeout','failed_buy','option_loop','repeated_option_actions','format_invalid_actions','unexecutable_actions','steps']}
    by_seed={str(seed):{k:statistics.mean(x[k] for x in features if x['replicate_seed']==seed) for k in means} for seed in card['replicate_seeds']}
    out=dict(status='verified',tag=tag,at=now(),episodes=1536,unique_tasks=512,means=means,counts=counts,by_seed=by_seed,
        task_manifest_sha256=digest(ROOT/'task_manifest.json'),source_hashes=source_hashes,
        elapsed_seconds=complete['elapsed_seconds'],features=sorted(features,key=lambda x:(x['panel_index'],x['replicate_seed'])))
    write(job/'verification.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['features','source_hashes']},ensure_ascii=False),flush=True)
    return out


def compare():
    import numpy as np
    card=verify_frozen()
    reports={}
    for tag in card['evaluation_order']+card['reference_tags']:
        p=ROOT/'jobs'/tag/'verification.json'
        if p.exists():
            reports[tag]=read(p)
    contrasts=[]
    pairs=[('r2_150', tag) for tag in card['reference_tags']]
    for candidate,reference in pairs:
        if candidate not in reports or reference not in reports:
            continue
        result=dict(candidate=candidate,reference=reference,metrics={})
        rng=np.random.default_rng(20261109)
        indices=rng.integers(0,512,size=(10000,512))
        for key in ['success','score','timeout','steps']:
            a=np.array([x[key] for x in reports[candidate]['features']]).reshape(512,3).mean(axis=1)
            b=np.array([x[key] for x in reports[reference]['features']]).reshape(512,3).mean(axis=1)
            delta=a-b
            boots=delta[indices].mean(axis=1)
            ci=np.quantile(boots,[.025,.975])
            result['metrics'][key]=dict(difference=float(delta.mean()),ci95=[float(x) for x in ci])
        contrasts.append(result)
    summary=dict(at=now(),protocol_sha256=digest(ROOT/'experiment_card.json'),
        checkpoints={tag:{k:v for k,v in rec.items() if k not in ['features','source_hashes']} for tag,rec in reports.items()},
        paired_contrasts=contrasts,
        interpretation='Paired bootstrap of512task clusters after averaging3generation repeats. Single training seed; reused development panel; final150 is the fixed primary endpoint of a35round continuation from aligned115. No unseen-goal or training-seed confidence claim.')
    write(ROOT/'comparison.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag');p.add_argument('--compare',action='store_true');a=p.parse_args()
    if a.tag:verify(a.tag)
    if a.compare:compare()
