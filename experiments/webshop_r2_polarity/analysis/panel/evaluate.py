"""Paired checkpoint inference using native prompts, environment and rollout."""
import argparse
import copy
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import time
from common import ROOT, R2, SOURCE, digest, now, read, request_seed, resources, verify_frozen, write


def main(tag):
    card=verify_frozen()
    assert tag in card['evaluation_order'] or tag=='gigpo_150'
    inventory=resources()
    model=ROOT/'models'/tag/'hf_model'
    merge=read(model.parent/'merge_verification.json')
    if digest(model/'model.safetensors')!=merge['model_sha256']:
        raise ValueError('Merged model digest changed')
    job=ROOT/'jobs'/tag
    job.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    write(job/'attempt.json',dict(at=now(),tag=tag,resources=inventory,card_sha256=digest(ROOT/'experiment_card.json'),merge=merge))
    import numpy as np
    import torch
    import ray
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl import DataProto
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
    from agent_system.multi_turn_rollout import TrajectoryCollector
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.environments.env_package.webshop import webshop_projection
    from panel_env import PanelBackend
    from recording import make_recorded_manager

    for cls in [TrajectoryCollector,WebshopEnvironmentManager,vLLMRollout]:
        assert Path(inspect.getfile(cls)).resolve().is_relative_to(SOURCE.resolve())
    config=OmegaConf.load(R2/'config.json')
    assert config.env.max_steps==15 and config.env.history_length==2 and config.data.val_batch_size==128
    assert config.data.max_prompt_length==4096 and config.data.max_response_length==512
    assert config.actor_rollout_ref.rollout.val_kwargs.temperature==0.4
    assert config.actor_rollout_ref.rollout.val_kwargs.do_sample
    tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=False)
    original=AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path,trust_remote_code=False)
    assert tokenizer.get_vocab()==original.get_vocab() and tokenizer.chat_template==original.chat_template
    assert tokenizer.pad_token_id==original.pad_token_id and tokenizer.eos_token_id==original.eos_token_id
    write(job/'preflight.json',dict(tag=tag,at=now(),tokenizer_equal=True,source_root=str(SOURCE),
        model_sha256=merge['model_sha256'],versions={p:importlib.metadata.version(p) for p in ['torch','vllm','ray','transformers']}))

    class SeededEngine:
        def __init__(self, engine):
            self.engine=engine
        def assign(self,tasks,replicate):
            self.tasks=tasks;self.replicate=replicate;self.turn=0
        def generate(self,prompts,sampling_params,**kwargs):
            assert len(prompts)==len(self.tasks)==128
            assert sampling_params.n==1 and sampling_params.temperature==0.4
            assert sampling_params.top_p==1.0 and sampling_params.top_k==-1
            params=[]
            for task in self.tasks:
                p=copy.deepcopy(sampling_params)
                p.seed=request_seed(task['task_uid'],self.replicate,self.turn)
                params.append(p)
            self.turn+=1
            return self.engine.generate(prompts=prompts,sampling_params=params,**kwargs)

    class EvaluationRollout(vLLMRollout):
        world_size=1
        def __init__(self):
            self.config=config.actor_rollout_ref.rollout
            self.pad_token_id=tokenizer.pad_token_id
            self.lora_kwargs={}
            engine=LLM(model=str(model),tokenizer=str(model),tensor_parallel_size=2,
                distributed_executor_backend='mp',dtype='bfloat16',enforce_eager=False,
                gpu_memory_utilization=0.6,disable_custom_all_reduce=True,
                max_model_len=4608,max_num_seqs=128,load_format='safetensors',
                disable_log_stats=True,max_num_batched_tokens=8192,enable_chunked_prefill=False,
                enable_prefix_caching=True,trust_remote_code=False,seed=0)
            self.inference_engine=SeededEngine(engine)
            kwargs=dict(n=1,logprobs=0,max_tokens=512,detokenize=False)
            for k in self.config:
                if hasattr(SamplingParams(),str(k)):
                    kwargs[k]=self.config.get(k)
            self.sampling_params=SamplingParams(**kwargs)
    assert EvaluationRollout.generate_sequences is vLLMRollout.generate_sequences

    RecordedManager=make_recorded_manager(WebshopEnvironmentManager)

    rollout=EvaluationRollout()
    runtime=OmegaConf.to_container(config.ray_init.runtime_env,resolve=True)
    runtime['env_vars']['PYTHONPATH']=os.environ['PYTHONPATH']
    runtime['env_vars']['CUDA_VISIBLE_DEVICES']=''
    ray.init(address='local',num_cpus=16,num_gpus=0,runtime_env=runtime,
             object_store_memory=1<<30,include_dashboard=False,
             _temp_dir='/tmp/ws-polarity-panel-'+tag+'-20261111',log_to_driver=False)
    backend=None
    all_stats=[]
    try:
        collector=TrajectoryCollector(config,tokenizer)
        backend=PanelBackend(config,SOURCE)
        manager=RecordedManager(backend,webshop_projection,config)
        manifest=read(ROOT/'task_manifest.json')['tasks']
        for replicate in card['replicate_seeds']:
            for batch_index in range(4):
                tasks=manifest[batch_index*128:(batch_index+1)*128]
                backend.assign(tasks)
                rollout.inference_engine.assign(tasks,replicate)
                raw=np.empty(128,dtype=object)
                raw[:]=[[{'role':'user','content':''}] for _ in range(128)]
                gen=DataProto.from_single_dict(dict(input_ids=torch.zeros((128,1),dtype=torch.long),
                    raw_prompt=raw,data_source=np.array(['webshop']*128,dtype=object)))
                gen.meta_info=dict(eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,
                    recompute_log_prob=False,do_sample=True,validate=True)
                started=time.monotonic()
                rows,rewards,lengths,success,uids,_=collector.vanilla_multi_turn_loop(
                    gen_batch=gen,actor_rollout_wg=rollout,envs=manager)
                records=[]
                for i,task in enumerate(tasks):
                    indices=[j for j,row in enumerate(rows[i]) if bool(row['active_masks'])]
                    steps=[dict(manager.trace_steps[i][j],turn=j,
                          sampling_seed=request_seed(task['task_uid'],replicate,j)) for j in indices]
                    assert len(steps)==int(lengths[i]) and 1<=len(steps)<=15
                    assert manager.tasks[i]==task['instruction']
                    won=bool(success['success_rate'][i]);score=float(success['webshop_task_score (not success_rate)'][i])
                    assert won==steps[-1]['won']==bool(rewards[i]>0)
                    assert score==steps[-1]['task_score'] and won==(score==1.0)
                    assert manager.initial_signatures[i]==task['initial_signature']
                    record=dict(tag=tag,checkpoint_step=merge['checkpoint']['step'],task_uid=task['task_uid'],
                        panel_index=task['panel_index'],replicate_seed=replicate,batch=batch_index,
                        episode_uid=str(uids[i]),instruction=task['instruction'],initial_signature=manager.initial_signatures[i],
                        goal_hash=task['goal_hash'],worker_seed=task['worker_seed'],goal_index=task['goal_index'],price_hash=task['price_hash'],
                        success=won,task_score=score,episode_reward=float(rewards[i]),steps=steps)
                    records.append(record)
                path=job/f'seed_{replicate}_batch_{batch_index}.jsonl'
                with path.open('x',encoding='utf-8') as f:
                    for rec in records:
                        f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                    f.flush();os.fsync(f.fileno())
                stats=dict(replicate=replicate,batch=batch_index,n=128,wins=sum(r['success'] for r in records),
                    score_sum=sum(r['task_score'] for r in records),timeouts=sum(not r['steps'][-1]['environment_done'] for r in records),
                    seconds=time.monotonic()-started,sha256=digest(path))
                all_stats.append(stats)
                progress=dict(tag=tag,at=now(),completed_episodes=128*len(all_stats),planned_episodes=1536,
                    batches=all_stats,elapsed_seconds=time.monotonic()-start)
                write(job/'progress.json',progress)
                print(json.dumps(dict(event='batch_complete',tag=tag,**stats)),flush=True)
                del rows,gen,records
        assert len(all_stats)==12
        write(job/'execution_complete.json',dict(tag=tag,at=now(),elapsed_seconds=time.monotonic()-start,batches=all_stats,
              interpretation='Execution complete; independent verify.py recount required.'))
    except BaseException as exc:
        write(job/'failure.json',dict(tag=tag,at=now(),error=repr(exc),elapsed_seconds=time.monotonic()-start))
        raise
    finally:
        if backend is not None:
            backend.close()
        ray.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    main(p.parse_args().tag)
