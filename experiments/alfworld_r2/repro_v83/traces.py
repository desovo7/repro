"""Adapt saved evaluation or original-training traces into the same tracker."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import time

from .tracker import Task,TrajectoryStateTracker

TASK_PRELOAD_RECORDS=[]


@lru_cache(maxsize=8192)
def task_from_path(path):
    return Task.from_gamefile(path)


def preload_task_metadata(gamefiles, *, split):
    """Parse the complete native task pool once during environment setup.

    Only goal metadata is cached. Observations, state transitions, action credit
    and outcomes are still computed for every original rollout at each step.
    The separate startup cost is persisted by RuntimeV83.
    """
    started=time.perf_counter()
    paths=tuple(str(path) for path in gamefiles)
    if not paths or len(set(paths))!=len(paths):raise ValueError('Invalid native task pool')
    before=task_from_path.cache_info()
    for path in paths:task_from_path(path)
    after=task_from_path.cache_info()
    record=dict(split=split,tasks=len(paths),new_metadata_reads=after.misses-before.misses,
                cache_hits=after.hits-before.hits,seconds=time.perf_counter()-started,
                ordered_paths_sha256=hashlib.sha256('\n'.join(paths).encode()).hexdigest())
    TASK_PRELOAD_RECORDS.append(record)
    return record


def projected_action(response):
    lower=response.lower()
    start,end=lower.find('<action>'),lower.find('</action>')
    return lower[-30:] if start<0 or end<0 else lower[start+8:end].strip()


@dataclass(slots=True)
class TrackedTrace:
    task: Task
    task_uid: str
    group_uid: str
    trajectory_uid: str
    success: bool
    complete: bool
    steps: list
    transitions: list
    source: dict


def track_trace(raw, *, prelit_bonus=0.1):
    task_uid=raw.get('task_uid',raw.get('game_id'))
    task=task_from_path(raw.get('gamefile',task_uid))
    tracker=TrajectoryStateTracker(task,prelit_bonus=prelit_bonus)
    evaluation=isinstance(raw.get('trajectory'),list)
    source_steps=raw['trajectory'] if evaluation else raw['steps']
    trajectory_uid=str(raw.get('trajectory_uid',raw.get('trace_id',raw.get('failure_id',task_uid))))
    group_uid=str(raw.get('group_uid',task_uid))
    tracker.reset(source_steps[0]['observation' if evaluation else 'observation_before'])
    steps,transitions=[],[]
    for i,s in enumerate(source_steps):
        if evaluation:
            obs=s['observation']
            after=s.get('next_observation',s.get('observation_after'))
            if after is None and i+1<len(source_steps):after=source_steps[i+1]['observation']
            action=s.get('action',projected_action(s['response']))
            won=bool(s['won']);valid=bool(s['format_valid'])
        else:
            obs=s['observation_before'];after=s['observation_after'];action=s['action']
            won=bool(s.get('won',s.get('reward_rel',False)))
            valid=bool(s.get('format_valid',s.get('official_is_action_valid',True)))
        if obs!=tracker.state.observation:
            raise ValueError(f'Observation chain discontinuity in {trajectory_uid} step {i}')
        transition=tracker.step(action,after,won=won,format_valid=valid)
        step=dict(action_uid=s.get('action_uid',f'{trajectory_uid}:{i}'),step_id=i,action=action,
                  observation_before=obs,observation_after=after,won=won,format_valid=valid,
                  environment_reward=float(s.get('official_environment_reward',s.get('reward',10*won))))
        if 'response_mask' in s:step['token_count']=sum(s['response_mask'])
        elif 'token_count' in s:step['token_count']=s['token_count']
        steps.append(step);transitions.append(transition)
    success=bool(raw.get('success',raw.get('won',False)))
    if transitions[-1].after.won!=success:raise ValueError('Final success and step won disagree')
    complete=success or len(steps)==50 or raw.get('end_reason')=='failure'
    return TrackedTrace(task,str(task_uid),group_uid,trajectory_uid,success,complete,steps,transitions,raw)
