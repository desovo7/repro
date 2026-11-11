"""Merge the frozen aligned checkpoints using the unchanged official merger."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT,R2,GIGPO,SOURCE,checkpoint_info,digest,now,read,verify_frozen,write

def merge(tag):
    card = verify_frozen()
    spec = card['checkpoints'].get(tag)
    if tag=='gigpo_150' and spec is None:
        spec = checkpoint_info(GIGPO,150)
        write(ROOT/'gigpo_150_checkpoint_input.json',spec)
    if spec is None:
        raise ValueError(tag)
    current = checkpoint_info(R2 if tag.startswith('r2_') else GIGPO,spec['step'])
    if current != spec:
        raise ValueError('Checkpoint input changed')
    target = ROOT/'models'/tag
    proof = target/'merge_verification.json'
    if proof.exists():
        done=read(proof)
        if digest(target/'hf_model/model.safetensors') != done['model_sha256']:
            raise ValueError('Merged model changed')
        return
    target.mkdir(parents=True,exist_ok=True)
    if (target/'hf_model').exists():
        raise FileExistsError('Incomplete previous merge requires investigation')
    start=time.monotonic()
    common=['--backend','fsdp','--local_dir',str(Path(spec['folder'])/'actor')]
    commands=[[sys.executable,str(SOURCE/'scripts/model_merger.py'),'merge',*common,'--target_dir',str(target/'hf_model')],
              [sys.executable,str(SOURCE/'scripts/model_merger.py'),'test',*common,'--test_hf_dir',str(target/'hf_model')]]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    for command,name in zip(commands,['merge.log','tensor_test.log'],strict=True):
        with (target/name).open('x',encoding='utf-8') as log:
            subprocess.run(command,env=env,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(proof,dict(tag=tag,status='passed',at=now(),checkpoint=spec,seconds=time.monotonic()-start,
        model_sha256=digest(target/'hf_model/model.safetensors'),commands=commands,
        verification='Official FSDP merge and full tensor comparison against both original shards; CUDA hidden.'))
    print('MERGE_VERIFIED '+tag,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    merge(p.parse_args().tag)
