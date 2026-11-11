"""Observe the unmodified rollout; annotate only its existing gather rows."""
from pathlib import Path


class ObservationCapture:
    def __init__(self,native):
        self.native=native

    def __getattr__(self,name):return getattr(self.native,name)

    def reset(self,*args,**kwargs):
        result=self.native.reset(*args,**kwargs)
        obs,infos=result
        self.initial_observations=list(obs['anchor'])
        self.gamefiles=[str(info['extra.gamefile']) for info in infos]
        self.steps=[]
        return result

    def step(self,text_actions,*args,**kwargs):
        # Native projection mutates text_actions in place; preserve generated
        # text before delegation and read the actual executed action afterward.
        generated=list(text_actions)
        result=self.native.step(text_actions,*args,**kwargs)
        obs,rewards,dones,infos=result
        records=[]
        for i,info in enumerate(infos):
            records.append(dict(observation_after=obs['anchor'][i],action=self.native.memory[i][-1]['action'],
                generated_text=generated[i],won=bool(info['won']),format_valid=bool(info['is_action_valid']),
                environment_done=bool(dones[i]),environment_reward=float(rewards[i])))
        self.steps.append(records)
        return result


def captured_traces(total_batch_list,capture,context,is_train):
    traces=[]
    for env_index,rows in enumerate(total_batch_list):
        active=[(time,row) for time,row in enumerate(rows) if bool(row['active_masks'])]
        if not active:raise ValueError('Original rollout has no active decision')
        first=active[0][1];gamefile=capture.gamefiles[env_index]
        split=next((x for x in reversed(Path(gamefile).parts) if x in {'train','valid_seen','valid_unseen'}),None)
        if split is None or (split=='train')!=is_train:raise ValueError('Actual task split disagrees with rollout mode')
        trace=dict(context,task_uid='/'.join(Path(gamefile).parts[-3:-1]),gamefile=gamefile,split=split,
                   group_uid=str(first['uid']),trajectory_uid=str(first['traj_uid']),steps=[])
        observation=capture.initial_observations[env_index]
        for step_id,(time,row) in enumerate(active):
            if str(row['uid'])!=trace['group_uid'] or str(row['traj_uid'])!=trace['trajectory_uid']:
                raise ValueError('Native gather row identity changed within a trajectory')
            if row['anchor_obs']!=observation:raise ValueError('Captured feedback and native anchor disagree')
            uid=f"{context['run_id']}:{context['iteration_id']}:{trace['trajectory_uid']}:{step_id}"
            row['repro_action_uid']=uid
            record=dict(capture.steps[time][env_index],action_uid=uid,step_id=step_id,
                        observation_before=observation)
            # This exact vector-environment reward remains the native reward.
            if float(row['rewards'])!=record['environment_reward']:raise ValueError('Reward capture mismatch')
            record['official_environment_reward']=record.pop('environment_reward')
            observation=record['observation_after'];trace['steps'].append(record)
        last=trace['steps'][-1]
        trace['success']=last['won']
        trace['end_reason']='success' if last['won'] else 'horizon' if len(active)==50 else 'failure' if last['environment_done'] else 'collector_truncation'
        traces.append(trace)
    return traces


def make_collector(native_collector):
    class CollectorV83(native_collector):
        v83_runtime=None

        def multi_turn_loop(self,*args,**kwargs):
            if self.v83_runtime is None:return super().multi_turn_loop(*args,**kwargs)
            if args or kwargs.get('rollout_mode','vanilla')!='vanilla':raise ValueError('Expected native vanilla keyword rollout')
            train=bool(kwargs.get('is_train',True));runtime=self.v83_runtime
            iteration=self.v83_trainer.global_steps
            context=runtime.start_iteration(iteration) if train else runtime.context(iteration)
            capture=ObservationCapture(kwargs['envs'])
            self._v83_pending=(capture,context,train)
            try:return super().multi_turn_loop(**dict(kwargs,envs=capture))
            finally:self._v83_pending=None

        def gather_rollout_data(self,total_batch_list,*args,**kwargs):
            pending=getattr(self,'_v83_pending',None)
            if pending:
                capture,context,train=pending
                traces=captured_traces(total_batch_list,capture,context,train)
                if train:
                    keys=self.v83_runtime.after_collection(traces)
                    for rows in total_batch_list:
                        for row in rows:
                            if bool(row['active_masks']):
                                row['v83_original_anchor_obs']=row['anchor_obs']
                                row['anchor_obs']=keys[row['repro_action_uid']]
                else:self.v83_runtime.record_evaluation(traces,context)
            return super().gather_rollout_data(total_batch_list,*args,**kwargs)

    return CollectorV83
