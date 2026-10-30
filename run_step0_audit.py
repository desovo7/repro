"""Full original-rollout audit, CPU only, with an explicit sign-flip estimand.

The estimand is strict positive<->negative change in native step-relative
advantage after de-aliasing. Zero-to-nonzero and nonzero-to-zero are separate.
No value labels are read from future iterations and no new rollouts are sampled.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time

try:
    from orjson import loads
except ImportError:
    loads=json.loads

from repro_v83.anchors import anchor_keys,sensitive_location,quantiles
from repro_v83.credit import group_membership,centered_auxiliary
from repro_v83.traces import track_trace


def sign(x):return (x>1e-10)-(x< -1e-10)


def compact_trace(raw):
    """Discard unused privileged graph fields after parsing the original row.

    Every original trajectory/action, observation, reward and identity remains.
    This does not subsample or simplify the state/credit algorithm.
    """
    result={k:v for k,v in raw.items() if k!='steps'}
    result['steps']=[]
    keys=('step_id','action_uid','action','observation_before','observation_after',
          'reward_rel','environment_done','terminated','official_environment_reward',
          'official_is_action_valid','won','format_valid')
    for step in raw['steps']:
        record={k:step[k] for k in keys if k in step}
        record['token_count']=sum(step['response_mask']) if 'response_mask' in step else step.get('token_count',0)
        result['steps'].append(record)
    return result


def mean_relative(rows,key):
    groups=defaultdict(list)
    for row in rows:groups[key(row)].append(row)
    output={}
    for group in groups.values():
        mean=sum(r['return'] for r in group)/len(group)
        # Subtracting the mean is sufficient for sign; std is positive.
        for row in group:output[row['uid']]=row['return']-mean
    return output


def audit_iteration(raw, gamma):
    t0=time.perf_counter()
    traces=[track_trace(r) for r in raw]
    _,af=group_membership(traces,8)
    keys,stats=anchor_keys(traces,af)
    aux=centered_auxiliary(traces)
    algorithm_seconds=time.perf_counter()-t0
    rows=[]
    for trace in traces:
        returns=[0.]*len(trace.steps);value=0.
        for i in reversed(range(len(returns))):
            value=trace.steps[i]['environment_reward']+gamma*value
            returns[i]=value
        if af[trace.group_uid]:continue
        for step,tr,value in zip(trace.steps,trace.transitions,returns):
            rows.append(dict(uid=step['action_uid'],group=trace.group_uid,return_=value,
                             native=step['observation_before'],new=keys[step['action_uid']],
                             eligible=('you arrive at ' in step['observation_before'].lower()
                                       and tr.before.held_obj is None and sensitive_location(trace.task,tr.before))))
    for row in rows:row['return']=row.pop('return_')
    native=mean_relative(rows,lambda r:(r['group'],r['native']))
    modified=mean_relative(rows,lambda r:(r['group'],r['new']))
    counts=Counter();examples=[]
    sizes={'native':Counter(),'dealiased':Counter()}
    for row in rows:
        sizes['native'][(row['group'],row['native'])]+=1
        sizes['dealiased'][(row['group'],row['new'])]+=1
        if row['eligible']:
            a,b=sign(native[row['uid']]),sign(modified[row['uid']])
            counts['eligible_empty_handed_arrival_actions']+=1
            counts['native_nonzero']+=a!=0
            counts['strict_sign_flips']+=a*b<0
            counts['zero_to_nonzero']+=a==0 and b!=0
            counts['nonzero_to_zero']+=a!=0 and b==0
            if a*b<0 and len(examples)<10:
                examples.append(dict(action_uid=row['uid'],native_centered_return=native[row['uid']],
                                     dealiased_centered_return=modified[row['uid']],observation=row['native']))
    return dict(iteration_id=raw[0]['iteration_id'],trajectories=len(traces),actions=sum(len(t.steps) for t in traces),
                groups=len(af),af_groups=sum(af.values()),anchor_stats=stats,sign_flip_counts=dict(counts),
                sign_flip_examples=examples,auxiliary=aux['diagnostics'],algorithm_seconds=algorithm_seconds,
                group_size_histograms={k:dict(Counter(v.values())) for k,v in sizes.items()})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--training-log',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native-gamma',type=float,default=.95)
    parser.add_argument('--compact-cache',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report_path=args.output/'step0_training_audit.json'
    if report_path.exists():raise FileExistsError('Preserve completed audits; use a fresh output directory')
    summaries=[];current=[];iteration=None;digest=hashlib.sha256();count=0;start=time.monotonic()
    source_root=Path(__file__).resolve().parent
    code_sha256={str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [source_root/'run_step0_audit.py',*sorted((source_root/'repro_v83').glob('*.py'))]}
    cache=args.compact_cache.open('x') if args.compact_cache else None
    with args.training_log.open('rb') as f:
        for line in f:
            digest.update(line)
            # JSON dictionaries/lists are acyclic; reference counting releases
            # the large old graph immediately. Do not let it pollute the small
            # state-tracker benchmark's cyclic-GC generations.
            gc_was_enabled=gc.isenabled()
            if gc_was_enabled:gc.disable()
            try:raw=compact_trace(loads(line))
            finally:
                if gc_was_enabled:gc.enable()
            if cache:cache.write(json.dumps(raw,separators=(',',':'))+'\n')
            if raw['split']!='train':raise ValueError('Unexpected evaluation data in training audit')
            i=raw['iteration_id']
            if iteration is not None and i!=iteration:
                if i<=iteration:raise ValueError('Non-monotonic original iterations')
                result=audit_iteration(current,args.native_gamma);summaries.append(result)
                print(json.dumps(dict(event='iteration_audited',iteration=iteration,actions=result['actions'],
                                      algorithm_seconds=result['algorithm_seconds'],elapsed_s=time.monotonic()-start)),flush=True)
                (args.output/'step0_progress.json').write_text(json.dumps(dict(last_iteration=iteration,trajectories=count,bytes_read=f.tell(),total_bytes=args.training_log.stat().st_size))+'\n')
                current=[]
            iteration=i;current.append(raw);count+=1
        if current:summaries.append(audit_iteration(current,args.native_gamma))
    if cache:cache.close()
    totals=Counter();histograms={'native':Counter(),'dealiased':Counter()}
    for s in summaries:
        totals.update(s['sign_flip_counts'])
        for mode,hist in s['group_size_histograms'].items():histograms[mode].update({int(k):v for k,v in hist.items()})
    distributions={}
    for mode,hist in histograms.items():
        expanded={i:v for i,v in enumerate(size for size,n in hist.items() for _ in range(n))}
        distributions[mode]=quantiles(expanded)
    denominator=totals['eligible_empty_handed_arrival_actions']
    result=dict(status='complete',cpu_only=True,created_at=datetime.now(timezone.utc).isoformat(),
                source=str(args.training_log),source_sha256=digest.hexdigest(),source_bytes=args.training_log.stat().st_size,
                code_sha256=code_sha256,compact_cache=str(args.compact_cache) if args.compact_cache else None,
                coverage='All original training trajectories in the supplied log; no iteration subsampling',
                iterations=len(summaries),trajectories=count,actions=sum(x['actions'] for x in summaries),
                sign_flip_estimand='Strict positive-to-negative or negative-to-positive native step-relative advantage change under the v8.3 adaptive anchor, restricted to mixed groups and empty-handed arrivals at goal/tool receptacles; pre-duplication unique action rows',
                sign_flip_counts=dict(totals),sign_flip_rate=(totals['strict_sign_flips']/denominator if denominator else None),
                anchor_distributions=distributions,algorithm_cpu_seconds={'max':max(x['algorithm_seconds'] for x in summaries),
                'mean':sum(x['algorithm_seconds'] for x in summaries)/len(summaries)},
                wall_seconds=time.monotonic()-start,per_iteration=summaries)
    report_path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_iteration'},indent=2))


if __name__=='__main__':main()
