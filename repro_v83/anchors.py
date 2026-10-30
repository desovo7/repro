"""De-alias only mixed-group GiGPO anchors; never alter actor observations."""
from collections import Counter
import json
import statistics

from .tracker import norm_observation,object_class


def sensitive_location(task,state):
    return (state.loc is not None and
            (object_class(state.loc) in {task.goal_recep_type,task.tool_recep_type}
             or (task.tool_recep_type=='desklamp' and state.loc in state.seen_lamp_locs)))


def quantiles(counts):
    values=sorted(counts.values())
    if not values:return {'n':0,'p25':None,'p50':None,'singleton_count':0,'singleton_ratio':0.0}
    def percentile(q):
        index=(len(values)-1)*q;lo=int(index);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(index-lo)
    return {'n':len(values),'p25':percentile(.25),'p50':percentile(.5),
            'singleton_count':values.count(1),'singleton_ratio':values.count(1)/len(values)}


def anchor_keys(traces,af,*,enabled=True):
    native,full,fallback=Counter(),Counter(),Counter()
    rows=[]
    for trace in traces:
        if af[trace.group_uid]:
            for step in trace.steps:
                rows.append((step['action_uid'],trace.group_uid,step['observation_before'],None,None))
            continue
        for step,transition in zip(trace.steps,trace.transitions):
            obs=step['observation_before'];state=transition.before
            normalized=norm_observation(obs)
            sensitive=sensitive_location(trace.task,state)
            full_key=(normalized,state.p,state.nav_bit)
            fallback_key=(normalized,state.p,state.nav_bit if sensitive else None)
            rows.append((step['action_uid'],trace.group_uid,obs,full_key,fallback_key))
            native[(trace.group_uid,obs)]+=1
            full[(trace.group_uid,full_key)]+=1
            fallback[(trace.group_uid,fallback_key)]+=1
    fallback_used=bool(full) and statistics.median(full.values())<2
    output={};serialized={}
    for uid,group,obs,key,backup in rows:
        if enabled and not af[group]:
            value=backup if fallback_used else key
            if value not in serialized:
                serialized[value]=json.dumps(value,ensure_ascii=False,separators=(',',':'))
            output[uid]=serialized[value]
        else:output[uid]=obs
    return output,dict(native=quantiles(native),full=quantiles(full),fallback=quantiles(fallback),
                       selected=quantiles((fallback if fallback_used else full) if enabled else native),
                       fallback_used=bool(enabled and fallback_used),modified=bool(enabled),
                       size_unit='unique original decision rows per anchor, mixed groups only')
