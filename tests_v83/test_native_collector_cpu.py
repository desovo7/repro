"""CPU checks of the genuine native preprocessing/gather/collate/adjust path.

Fixture rows come from saved real ALFWorld rollouts. Generation and environment
execution are deliberately outside this test; no model or fake trainer runs.
"""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


def run_collector_probe(config, compact_path, output_dir):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from verl import DataProto
    from agent_system.multi_turn_rollout import TrajectoryCollector
    from agent_system.multi_turn_rollout.utils import adjust_batch
    from repro_v83.collector import make_collector
    from repro_v83.runtime import RuntimeV83

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Collector probe requires hidden GPUs')
    rows = json.loads(Path(compact_path).read_text())['rows']
    groups = {}
    for row in rows:
        groups.setdefault(row['group_uid'], []).append(row)
    mixed = next(g for g in groups.values() if any(t['success'] for t in g))
    failed = next(g for g in groups.values() if not any(t['success'] for t in g))
    selected = mixed + failed
    assert len(mixed) == len(failed) == 8
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path, local_files_only=True)
    native = TrajectoryCollector(config, tokenizer)
    seed = DataProto.from_dict(tensors={'input_ids':torch.zeros((1,1),dtype=torch.long)},
        non_tensors={'raw_prompt':np.array([[{'role':'user','content':'fixture'}]],dtype=object),
                     'data_source':np.array(['alfworld'],dtype=object)})
    pending = SimpleNamespace(initial_observations=[t['steps'][0]['observation_before'] for t in selected],
        gamefiles=[t['gamefile'] for t in selected], steps=[])
    batches = [[] for _ in selected]
    prompt_cache = {}
    response_cache = {}
    for time in range(max(len(t['steps']) for t in selected)):
        captured_step = []
        for index, trace in enumerate(selected):
            if time >= len(trace['steps']):
                captured_step.append({})
                continue
            step = trace['steps'][time]
            observation = step['observation_before']
            if observation not in prompt_cache:
                # Use the real native chat-template/tokenization path. The
                # saved compact record lacks full admissible-action prompts;
                # this is a controlled prompt fixture, not a reconstruction.
                prompt_cache[observation] = native.preprocess_single_sample(0, seed,
                    {'text':[observation], 'anchor':[observation]})
            row = copy.deepcopy(prompt_cache[observation])
            generated = '<think>Replay the saved action.</think><action>'+step['action']+'</action>'
            if generated not in response_cache:
                ids = tokenizer.encode(generated, add_special_tokens=False)
                limit = config.data.max_response_length
                assert len(ids) <= limit
                response_cache[generated] = (torch.tensor(ids+[tokenizer.pad_token_id]*(limit-len(ids))),
                    torch.tensor([1]*len(ids)+[0]*(limit-len(ids))))
            response, response_mask = response_cache[generated]
            prompts, attention = row['input_ids'], row['attention_mask']
            row.update(prompts=prompts, responses=response.clone(), response_mask=response_mask.clone(),
                input_ids=torch.cat((prompts,response)), attention_mask=torch.cat((attention,response_mask)),
                position_ids=torch.arange(len(prompts)+len(response)),
                uid=trace['group_uid'], traj_uid=trace['trajectory_uid'], active_masks=True,
                rewards=step['official_environment_reward'], is_action_valid=step['format_valid'])
            batches[index].append(row)
            captured_step.append(dict(observation_after=step['observation_after'],action=step['action'],
                generated_text=generated,won=step['won'],format_valid=step['format_valid'],
                environment_done=step['won'] or time+1==len(trace['steps']),
                environment_reward=step['official_environment_reward']))
        pending.steps.append(captured_step)
    config.repro_v83.output_dir = str(output_dir)
    runtime = RuntimeV83(config)
    context = runtime.start_iteration(1)
    augmented = make_collector(TrajectoryCollector)(config, tokenizer)
    augmented.v83_runtime = runtime
    augmented._v83_pending = (pending, context, True)
    kwargs = dict(episode_rewards=np.array([sum(s['official_environment_reward'] for s in t['steps']) for t in selected]),
        episode_lengths=np.array([len(t['steps']) for t in selected]),
        success={'success_rate':np.array([t['success'] for t in selected])},
        traj_uid=np.array([t['trajectory_uid'] for t in selected]), tool_callings=np.zeros(len(selected)))
    baseline = native.gather_rollout_data(copy.deepcopy(batches), **kwargs)
    candidate = augmented.gather_rollout_data(copy.deepcopy(batches), **kwargs)
    assert len(baseline) == len(candidate)
    tensor_checks = []
    for name in baseline.batch.keys():
        torch.testing.assert_close(baseline.batch[name], candidate.batch[name], rtol=0, atol=0)
        tensor_checks.append(name)
    non_tensor_checks = []
    for name, values in baseline.non_tensor_batch.items():
        if name != 'anchor_obs':
            assert values.tolist() == candidate.non_tensor_batch[name].tolist(), name
            non_tensor_checks.append(name)
    mixed_changed = failed_unchanged = 0
    lookup = {}
    for i, uid in enumerate(candidate.non_tensor_batch['repro_action_uid']):
        group = candidate.non_tensor_batch['uid'][i]
        original = baseline.non_tensor_batch['anchor_obs'][i]
        current = candidate.non_tensor_batch['anchor_obs'][i]
        assert candidate.non_tensor_batch['v83_original_anchor_obs'][i] == original
        if runtime.credits['af'][group]:
            assert current == original
            failed_unchanged += 1
        else:
            assert current != original
            mixed_changed += 1
        lookup[uid] = i
    np.random.seed(0)
    adjusted = adjust_batch(config, candidate, mode='copy')
    assert len(adjusted) > len(candidate), 'Fixture should exercise actual copy adjustment'
    copied = len(adjusted)-len(candidate)
    adjusted.reorder(torch.arange(len(adjusted)-1,-1,-1))
    aligned = 0
    for i, uid in enumerate(adjusted.non_tensor_batch['repro_action_uid']):
        source = lookup[uid]
        for key in ['input_ids','responses','attention_mask','response_mask']:
            assert torch.equal(adjusted.batch[key][i], candidate.batch[key][source]), key
        for key in ['uid','traj_uid','anchor_obs','v83_original_anchor_obs','rewards','raw_prompt']:
            v0,v1=adjusted.non_tensor_batch[key][i],candidate.non_tensor_batch[key][source]
            if isinstance(v0,np.ndarray):assert v0.tolist()==v1.tolist(),key
            else:assert v0==v1,key
        aligned += 1
    assert not torch.cuda.is_initialized()
    return dict(status='passed',native_collector_module=TrajectoryCollector.__module__,
        tokenizer_class=type(tokenizer).__name__,fixture_trajectories=16,group_size=8,
        original_rows=len(candidate),mixed_anchor_rows_changed=mixed_changed,af_anchor_rows_unchanged=failed_unchanged,
        all_native_tensor_fields_unchanged=tensor_checks,all_other_native_non_tensor_fields_unchanged=non_tensor_checks,
        actual_adjust_batch_copies=copied,rows_verified_after_native_copy_and_reorder=aligned,
        native_prompt_limit=config.data.max_prompt_length,native_response_limit=config.data.max_response_length,
        cuda_initialized=torch.cuda.is_initialized(),limitations=[
            'Saved authentic environment observations/actions/rewards; no live environment execution in this gather test.',
            'Controlled prompts use actual native preprocessing and tokenizer; full historical actor prompts are unavailable.',
            'No model generation, advantage estimation, GPU worker or optimizer update is exercised.'])


@unittest.skipUnless(os.environ.get('V83_NATIVE_COLLECTOR_CPU')=='1','requires native runtime and supplied paths')
class NativeCollectorTests(unittest.TestCase):
    def test_gather_identity_and_native_copy_reorder(self):
        from omegaconf import OmegaConf
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            result=run_collector_probe(OmegaConf.load(os.environ['V83_NATIVE_CONFIG']),
                                       os.environ['V83_COMPACT_TRACES'],tmp)
            self.assertEqual(result['status'],'passed')


if __name__=='__main__':unittest.main()
