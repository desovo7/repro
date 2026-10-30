"""Transactional iteration lifecycle and actual-response-token credit injection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
import uuid

from repro_external.common import atomic_json,value
from .anchors import anchor_keys
from .credit import CreditConfig,centered_auxiliary,weighted_std
from .traces import track_trace,TASK_PRELOAD_RECORDS


class RuntimeV83:
    def __init__(self,config):
        self.config=config
        self.options=value(config,'repro_v83')
        self.run_id=value(config,'repro_v83.run_id')
        self.config_hash=value(config,'repro_v83.config_hash')
        self.model_contract_hash=value(config,'repro_v83.model_contract_hash')
        self.output=Path(value(config,'repro_v83.output_dir'))
        self.output.mkdir(parents=True,exist_ok=True)
        self.tier1a=bool(value(config,'repro_v83.enable_tier1a',False))
        self.tier1b=bool(value(config,'repro_v83.enable_tier1b',False))
        self.tier2=bool(value(config,'repro_v83.enable_tier2',False))
        if self.tier1b and not self.tier1a:raise ValueError('Tier 1B requires Tier 1A')
        if self.tier2 and not self.tier1b:raise ValueError('Tier 2 requires Tier 1A and Tier 1B')
        self.tier2_pool_mode=value(config,'repro_v83.tier2_pool_mode','mixed_only')
        self.value_table=None;self.value_snapshot=None
        if self.tier2:
            from .value_table import QuotientValueTable
            self.value_table=QuotientValueTable(pool_mode=self.tier2_pool_mode,window=4,alpha0=5.0)
        if int(value(config,'env.max_steps'))!=50:raise ValueError('v8.3 is frozen to the 50-step protocol')
        if value(config,'algorithm.gigpo.enable_similarity',False):raise ValueError('v8.3 requires exact native anchor grouping')
        if not self.config_hash or not self.model_contract_hash:raise ValueError('Run contracts must be frozen')
        self.credit_config=CreditConfig(**dict(value(config,'repro_v83.credit',{})))
        self.group_size=int(value(config,'env.rollout.n'))
        if self.group_size!=8:raise ValueError('v8.3 requires exactly eight original trajectories per task group')
        atomic_json(self.output/'task_metadata_preload.json',dict(
            records=list(TASK_PRELOAD_RECORDS),
            note='One-time native full-pool goal-metadata parsing; charged separately from per-iteration state/credit calculations'))
        self.phase='idle';self.iteration_id=None;self.optimizer_steps=0

    def start_iteration(self,iteration):
        if self.phase not in {'idle','committed'}:raise RuntimeError('Prior iteration still pending')
        self.iteration_id=int(iteration)
        self.snapshot_id=f'{self.config_hash}:iteration-{self.iteration_id}'
        if self.tier2:
            started=time.perf_counter()
            self.value_snapshot=self.value_table.snapshot(self.iteration_id)
            self.table_snapshot_seconds=time.perf_counter()-started
            atomic_json(self.output/'value_snapshots'/f'iteration_{self.iteration_id}.json',dict(
                context=self.context(iteration),diagnostics=self.value_snapshot.diagnostics))
        self.phase='collecting'
        return self.context(iteration)

    def context(self,iteration):
        return dict(run_id=self.run_id,iteration_id=int(iteration))

    def after_collection(self,raw_traces):
        if self.phase!='collecting':raise RuntimeError('Collection lacks a matching iteration')
        started=time.perf_counter()
        if self.tier2 and any(t.get('split')!='train' or t.get('run_id')!=self.run_id
                             or t.get('iteration_id')!=self.iteration_id for t in raw_traces):
            raise ValueError('Tier 2 collection must belong to the current training run and iteration')
        traces=[track_trace(t) for t in raw_traces]
        self.current=traces
        self.credits=centered_auxiliary(traces,self.credit_config,expected_group_size=self.group_size,enabled=self.tier1b,
            value_provider=self.value_snapshot.values_for_trace if self.tier2 else None)
        self.keys,self.anchor_stats=anchor_keys(traces,self.credits['af'],enabled=self.tier1a)
        self.action_groups={step['action_uid']:t.group_uid for t in traces for step in t.steps}
        self.algorithm_seconds=time.perf_counter()-started
        self.phase='collected'
        # Keep complete observations/responses for every original action, but
        # no privileged fact graph or redundant historical snapshots.
        with (self.output/'train_traces.jsonl').open('a') as f:
            for raw in raw_traces:f.write(json.dumps(raw,ensure_ascii=False,allow_nan=False)+'\n')
        return self.keys

    def record_evaluation(self,traces,context):
        path=self.output/'eval_traces'/f"iteration_{context['iteration_id']}.{uuid.uuid4().hex}.jsonl"
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(t,ensure_ascii=False,allow_nan=False)+'\n' for t in traces))

    def inject(self,batch):
        if self.phase!='collected':raise RuntimeError('Actor injection out of order')
        import torch
        started=time.perf_counter()
        base=batch.batch['advantages'];mask=batch.batch['response_mask']
        if (base.shape!=mask.shape or base.ndim!=2 or not torch.isfinite(base).all()
            or not torch.isfinite(mask).all() or not torch.all((mask==0)|(mask==1))):
            raise ValueError('Native actor advantage shape or finite-value check failed')
        if batch.meta_info.get('repro_credit_applied'):raise RuntimeError('Duplicate auxiliary injection')
        ids=list(batch.non_tensor_batch['repro_action_uid'])
        if len(ids)!=base.shape[0]:raise ValueError('Actor action identities and rows disagree')
        token_counts=mask.sum(dim=-1).detach().cpu().tolist()
        values=[];af_values=[];af_weights=[]
        for uid,weight in zip(ids,token_counts):
            if uid not in self.action_groups:raise ValueError('Unknown actor row action identity')
            val=self.credits['centered'][uid] if self.tier1b else 0.
            values.append(val)
            if self.credits['af'][self.action_groups[uid]]:
                af_values.append(val);af_weights.append(weight)
        sigma=weighted_std(af_values,af_weights)
        scale=self.credit_config.beta/(sigma+self.credit_config.epsilon) if self.tier1b else 0.
        delta=torch.tensor(values,dtype=base.dtype,device=base.device)[:,None]*scale*mask.to(base.dtype)
        combined=base+delta
        if not torch.isfinite(combined).all():raise FloatingPointError('Nonfinite v8.3 injected advantage')
        batch.batch['advantages']=combined.detach()
        batch.batch['repro_aux_delta']=(combined-base).detach()
        batch.meta_info['repro_credit_applied']=True
        batch.meta_info['repro_external_audit_context']=dict(output_dir=str(self.output.resolve()),
            run_id=self.run_id,iteration_id=self.iteration_id,snapshot_id=self.snapshot_id)
        self.inject_seconds=time.perf_counter()-started
        self.injection=dict(af_token_std=sigma,af_response_tokens=sum(af_weights),scale=scale,
            nonzero_delta_tokens=int(torch.count_nonzero(delta)),max_abs_delta=float(delta.abs().max()),
            centered_actions_positive=sum(v>0 for v in values),centered_actions_negative=sum(v<0 for v in values))
        self.phase='injected'

    def commit(self,output):
        if self.phase!='injected':raise RuntimeError('Actor commit out of order')
        audit=output.meta_info.get('repro_external_audit',{})
        if any(audit.get(k)!=getattr(self,k) for k in ['run_id','iteration_id','snapshot_id']):
            raise RuntimeError('Actor audit belongs to another run or iteration')
        steps=audit.get('actual_steps_min',0)
        if (not audit.get('all_ranks_finite') or steps<=0 or
            any(audit.get(k)!=steps for k in ['actual_steps_max','expected_steps_min','expected_steps_max'])
            or audit.get('world_size')!=2):raise RuntimeError('Two-rank actor update verification failed')
        table_record={}
        if self.tier2:
            started=time.perf_counter()
            calibration=self.value_table.commit(self.iteration_id,self.current,self.credits['af'])
            self.table_commit_seconds=time.perf_counter()-started
            table_record=dict(snapshot=self.value_snapshot.diagnostics,current_iteration_calibration=calibration,
                              snapshot_seconds=self.table_snapshot_seconds,commit_seconds=self.table_commit_seconds)
        self.optimizer_steps+=steps
        metrics={**self.credits['diagnostics'],**self.injection,
                 'algorithm_cpu_seconds':self.algorithm_seconds,'injection_seconds':self.inject_seconds,
                 'anchor_fallback_used':int(self.anchor_stats['fallback_used']),
                 'anchor_singleton_ratio':self.anchor_stats['selected']['singleton_ratio'],
                 'anchor_full_singleton_ratio':self.anchor_stats['full']['singleton_ratio'],
                 'anchor_native_singleton_ratio':self.anchor_stats['native']['singleton_ratio']}
        if self.tier2:
            metrics.update(table_snapshot_seconds=self.table_snapshot_seconds,table_commit_seconds=self.table_commit_seconds)
            for name,values in [('support',self.value_snapshot.diagnostics),('calibration',calibration)]:
                if isinstance(values,dict):
                    metrics.update({'tier2_'+name+'_'+k:v for k,v in values.items()
                                    if isinstance(v,(int,float)) and not isinstance(v,bool)})
        output.meta_info['metrics'].update({'v83/'+k:v for k,v in metrics.items()})
        atomic_json(self.output/'diagnostics'/f'iteration_{self.iteration_id}.json',dict(
            context=self.context(self.iteration_id),snapshot_id=self.snapshot_id,audit=audit,
            credit=self.credits['diagnostics'],injection=self.injection,anchor_stats=self.anchor_stats,
            metrics=metrics,optimizer_steps_per_rank=self.optimizer_steps,tier2=table_record))
        self.phase='committed'

    def save(self,folder,*,task_cursors=None):
        if self.phase!='committed':raise RuntimeError('Cannot checkpoint a pending actor update')
        if (not isinstance(task_cursors,dict) or set(task_cursors)!={'train','validation'}
            or not task_cursors['train'] or not task_cursors['validation']):
            raise ValueError('Checkpoint requires every native training/validation task cursor')
        import numpy as np
        import torch
        folder=Path(folder)
        state=dict(run_id=self.run_id,config_hash=self.config_hash,model_contract_hash=self.model_contract_hash,
                   iteration_id=self.iteration_id,optimizer_steps=self.optimizer_steps,version='8.3.1',
                   task_cursors=task_cursors,enable_tier2=self.tier2)
        if self.tier2:
            state['tier2_state']=self.value_table.state_dict()
            if state['tier2_state']['last_committed_iteration']!=self.iteration_id:
                raise RuntimeError('Tier 2 history and optimizer checkpoint disagree')
        atomic_json(folder/'v83_runtime.json',state)
        rng=folder/'v83_driver_rng.pt';tmp=folder/'v83_driver_rng.pt.tmp'
        torch.save(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state()),tmp)
        tmp.replace(rng)
        atomic_json(folder/'V83_COMPLETE.json',dict(**state,
            runtime_sha256=hashlib.sha256((folder/'v83_runtime.json').read_bytes()).hexdigest(),
            rng_sha256=hashlib.sha256(rng.read_bytes()).hexdigest()))

    def load(self,folder):
        import numpy as np
        import torch
        folder=Path(folder)
        marker=json.loads((folder/'V83_COMPLETE.json').read_text())
        for key in ['run_id','config_hash','model_contract_hash']:
            if marker[key]!=getattr(self,key):raise ValueError('Checkpoint contract mismatch')
        for name,key in [('v83_runtime.json','runtime_sha256'),('v83_driver_rng.pt','rng_sha256')]:
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=marker[key]:raise ValueError('Checkpoint digest mismatch')
        state=json.loads((folder/'v83_runtime.json').read_text())
        if bool(state.get('enable_tier2',False))!=self.tier2:
            raise ValueError('Tier 2 checkpoint mode mismatch')
        if self.tier2:
            if state['tier2_state']['last_committed_iteration']!=marker['iteration_id']:
                raise ValueError('Tier 2 checkpoint history is incomplete')
            self.value_table.load_state_dict(state['tier2_state'])
            self.value_snapshot=None
        self.task_cursors=state['task_cursors']
        if folder.name!=f"global_step_{marker['iteration_id']}":raise ValueError('Native/runtime checkpoint mismatch')
        rng=torch.load(folder/'v83_driver_rng.pt',map_location='cpu',weights_only=False)
        random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
        self.iteration_id=marker['iteration_id'];self.optimizer_steps=marker['optimizer_steps'];self.phase='idle'
