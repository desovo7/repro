"""CPU-only v8.3.1 potential/table credit and truncated GAE calculations."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from .tracker import Task, Transition, is_search, prior_potential


@dataclass(frozen=True, slots=True)
class CreditConfig:
    gamma: float = 0.98
    lambda_aux: float = 0.5
    kappa_nav: float = 0.05
    noop_cost: float = 0.05
    beta: float = 0.5
    epsilon: float = 1e-6

    def __post_init__(self):
        if not (0 <= self.gamma <= 1 and 0 <= self.lambda_aux <= 1):
            raise ValueError('Discount and GAE decay must be in [0,1]')
        if not all(math.isfinite(v) and v >= 0 for v in (self.kappa_nav,self.noop_cost,self.beta)) or self.epsilon <= 0:
            raise ValueError('Invalid auxiliary credit coefficient')


def budget_bucket(step_id: int) -> int:
    """Zero-based decision index -> specification's [1,10]/[11,25]/[26,50]."""
    if not 0 <= step_id < 50:
        raise ValueError('v8.3 requires a 50-action horizon')
    return 0 if step_id < 10 else 1 if step_id < 25 else 2


def td_credit(task: Task, transition: Transition, config=CreditConfig(), *, values=None):
    if transition.feedback_missing and not transition.after.won:
        raise ValueError('Missing next observation cannot establish training credit')
    if transition.no_op:
        return -config.noop_cost, 0.0
    before,after=transition.before,transition.after
    if values is None:
        v0,v1=prior_potential(task,before),prior_potential(task,after)
        delta=config.gamma*v1-v0
        if before.p==after.p and is_search(task,before.p):
            delta+=config.kappa_nav*(2*after.nav_bit-1)
    else:
        # Tier 2 embeds navigation in the frozen table: do not add Tier 1B's
        # navigation shaping a second time. Terminal boundary is deterministic.
        v0,v1=values
        if not all(math.isfinite(v) for v in (v0,v1)):
            raise ValueError('Nonfinite Tier 2 state value')
        if after.won or after.p==task.K:v1=1.0
        delta=config.gamma*v1-v0
        if (task.family=='look_at_obj_in_light' and before.p<=1
                and transition.first_lamp_activation):
            delta+=0.10
    return delta,config.lambda_aux


def trajectory_gae(task, transitions, config=CreditConfig(), *, values=None):
    """A no-op cuts its own outgoing GAE edge, before any centering."""
    if values is not None:
        if len(values)!=len(transitions)+1 or not all(math.isfinite(v) for v in values):
            raise ValueError('Tier 2 requires one finite value for each trajectory state')
    deltas,decays=[],[]
    for i,t in enumerate(transitions):
        pair=None if values is None else tuple(values[i:i+2])
        if pair is not None and i==len(transitions)-1 and not (t.after.won or t.after.p==task.K):
            pair=(pair[0],0.0)  # completed unsuccessful episode boundary
        delta,decay=td_credit(task,t,config,values=pair)
        deltas.append(delta);decays.append(decay)
    result=[0.0]*len(transitions)
    tail=0.0
    for i in reversed(range(len(transitions))):
        tail=deltas[i]+config.gamma*decays[i]*tail
        result[i]=tail
    return result,deltas,decays


def group_membership(traces, expected_group_size=8):
    groups=defaultdict(list)
    seen=set()
    for trace in traces:
        if trace.trajectory_uid in seen:
            raise ValueError('Duplicate original trajectory before batch adjustment')
        seen.add(trace.trajectory_uid)
        groups[trace.group_uid].append(trace)
    flags={}
    for uid,group in groups.items():
        if len(group)!=expected_group_size:
            raise ValueError(f'Incomplete original task group {uid}: {len(group)}')
        if len({t.task_uid for t in group})!=1:
            raise ValueError('Task group contains more than one task')
        for trace in group:
            if not trace.success and not trace.complete:
                raise ValueError('A truncated unsuccessful rollout cannot be classified all-failed')
        flags[uid]=not any(t.success for t in group)
    return groups,flags


def centered_auxiliary(traces, config=CreditConfig(), *, expected_group_size=8, enabled=True, value_provider=None):
    """Center original action-level GAE within (task group, decision bucket).

    Token-weighted batch standard deviation is applied at the final actor
    boundary, after native batch copying/reordering, using its actual mask.
    """
    groups,af=group_membership(traces,expected_group_size)
    raw,centered,delta_by_uid={},{},{}
    buckets=defaultdict(list)
    counts={'groups':len(groups),'af_groups':sum(af.values()),'actions':0,'noop_actions':0,
            'forward_actions':0,'nonpositive_forward_td':0,'recovery_actions':0,'nonpositive_recovery_td':0,
            'optimal_search_actions':0,'nonpositive_optimal_search_td':0}
    for trace in traces:
        if enabled and af[trace.group_uid]:
            vals=None if value_provider is None else value_provider(trace)
            aux,deltas,_=trajectory_gae(trace.task,trace.transitions,config,values=vals)
        else:
            aux=deltas=[0.0]*len(trace.steps)
        for step,transition,a,delta in zip(trace.steps,trace.transitions,aux,deltas):
            uid=step['action_uid']
            if uid in raw:
                raise ValueError('Duplicate original action identity')
            raw[uid]=a;delta_by_uid[uid]=delta;centered[uid]=0.0
            counts['actions']+=1
            if enabled and af[trace.group_uid]:
                buckets[(trace.group_uid,budget_bucket(step['step_id']))].append(uid)
                counts['noop_actions']+=int(transition.no_op)
                if not transition.no_op:
                    before,after=transition.before,transition.after
                    if 0<=before.p<after.p:
                        counts['forward_actions']+=1
                        counts['nonpositive_forward_td']+=int(delta<=0)
                    if before.p==-1 and after.p>=0:
                        counts['recovery_actions']+=1
                        counts['nonpositive_recovery_td']+=int(delta<=0)
                    if before.p==after.p and is_search(trace.task,before.p) and after.nav_bit==1:
                        counts['optimal_search_actions']+=1
                        counts['nonpositive_optimal_search_td']+=int(delta<=0)
    zero_variance=0;max_mean_residual=0.0
    for members in buckets.values():
        values=center_within_bucket([raw[uid] for uid in members])
        if not any(values):zero_variance+=1
        for uid,v in zip(members,values):centered[uid]=v
        max_mean_residual=max(max_mean_residual,abs(math.fsum(values)/len(values)))
    counts.update(nonzero_centered_actions=sum(v!=0 for v in centered.values()),centered_buckets=len(buckets),
                  zero_variance_buckets=zero_variance,centering_max_abs_mean=max_mean_residual)
    return dict(centered=centered,raw=raw,deltas=delta_by_uid,af=af,diagnostics=counts)


def center_within_bucket(values):
    """Specification T2-2: action-weighted centering with a variance floor."""
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite auxiliary credit')
    if len(values)<2:return [0.0]*len(values)
    mean=sum(values)/len(values)
    centered=[v-mean for v in values]
    if math.fsum(v*v for v in centered)/len(centered)<=1e-12:
        return [0.0]*len(values)
    return centered


def weighted_std(values, weights):
    """Population std over the expanded response-token multiset, without expansion."""
    total=sum(weights)
    if total<=0:return 0.0
    mean=sum(v*w for v,w in zip(values,weights))/total
    return math.sqrt(max(0.,sum(w*(v-mean)**2 for v,w in zip(values,weights))/total))
