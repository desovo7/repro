"""Independent read-only audit of actual reload and final complete checkpoints."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path('/data_b/qzluo/workspace/wlx/Repro-verl-agent/runs/gradus-webshop-h800-s0-polarity-from115-20261111')
CONFIG_SHA = '7182f6c4ea4aeabcc5e564951294bc3454923e16ac15ec6c6a42b0ba248a80c6'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(final=False):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT))
    from webshop.native_retention import REQUIRED
    assert sha(ROOT/'config.json') == CONFIG_SHA
    config = read(ROOT/'config.json')
    for key, directory in [('files', 'webshop'), ('source', 'source'), ('data', 'data')]:
        for name, expected in config['webshop_release'][key].items():
            assert sha(ROOT/directory/name) == expected, name
    # Retention publishes before Ray forwards the per-step console metrics.
    for _ in range(30):
        log = (ROOT/'train.log').read_text()
        observed = [int(float(x)) for x in re.findall(r'training/global_step:([\d.]+)', log)]
        if observed and max(observed) >= 117:
            break
        time.sleep(1)
    assert 'NATIVE_RESUMED step=116 integrity_verified=True' in log
    completed = [int(float(x)) for x in re.findall(r'training/global_step:([\d.]+)', log)]
    assert max(completed) >= 117
    assert not any(n != 1 for n in Counter(completed).values()), 'Duplicated completed iterations'
    assert sorted(completed) == list(range(116, max(completed)+1))
    index = read(ROOT/'checkpoints/best_checkpoints/index.json')
    assert index['metric'] == 'val/success_rate' and index['keep_best'] == 3
    assert index['tie_break'] == 'later iteration first' and index['pinned_steps'] == [115, 116]
    expected = sorted((r for r in index['history'] if r['score'] is not None),
                      key=lambda r: (-r['score'], -r['step']))[:3]
    assert index['best'] == expected
    decisions = [read(p) for p in (ROOT/'checkpoints/best_checkpoints/resume_decisions').glob('*.json')]
    selected = [d for d in decisions if (d.get('selected') or {}).get('step') == 116]
    assert len(selected) == 1 and selected[0]['status'] == 'loaded'
    step = 150 if final else 117
    if final:
        assert index['latest']['step'] == max(completed) == 150
        assert 'Final validation metrics:' in log
        assert len(index['best']) == 3
        protected = {r['step'] for r in index['best']} | {150, 115, 116}
    else:
        protected = {117}
    # Audit every real actor update and its continuity across the restart.
    audited = []
    parent_steps = read(ROOT/'first_checkpoint_verified.json')['parent_optimizer_steps']
    previous = {rank: [parent_steps[rank][0]]*2 for rank in (0, 1)}
    for iteration in range(116, step+1):
        diag = read(ROOT/f'gradus/diagnostics/iteration_{iteration}.json')
        assert diag['metrics']['auxiliary_gated_nonzero_tokens'] == 0
        assert diag['audit']['all_ranks_finite'] and diag['audit']['world_size'] == 2
        rows = []
        for rank in (0, 1):
            row = read(ROOT/f'gradus/actor_audit/iteration_{iteration}.rank_{rank}.json')
            assert row['iteration_id'] == iteration and row['rank'] == rank
            assert row['before_steps'] == previous[rank]
            assert row['actual_steps'] == row['expected_steps'] > 0 and row['finite']
            assert row['after_steps'][0] == row['after_steps'][1]
            previous[rank] = row['after_steps']
            rows.append(row)
        assert rows[0]['after_steps'] == rows[1]['after_steps']
        audited.append(dict(iteration=iteration, before=rows[0]['before_steps'][0],
                            after=rows[0]['after_steps'][0], updates=rows[0]['actual_steps']))
    proofs = {}
    folders = {}
    for number in sorted(protected):
        folder = ROOT/f'checkpoints/best_checkpoints/global_step_{number}'
        if not final:
            folder = ROOT/f'checkpoints/global_step_{number}'
        folders[number] = folder
        marker, runtime, files = [read(folder/name) for name in
                                 ['NATIVE_COMPLETE.json', 'native_runtime.json', 'native_files.json']]
        assert all((folder/name).is_file() for name in REQUIRED)
        assert marker['iteration_id'] == runtime['iteration_id'] == number
        assert runtime['world_size'] == 2 and runtime['gradus_state']['last_iteration'] == number
        assert runtime['gradus_state']['schema'] == 'gradus-webshop-r2-polarity-v1'
        assert sha(folder/'native_files.json') == marker['checkpoint_manifest_sha256']
        assert sha(folder/'native_runtime.json') == marker['runtime_sha256']
        assert sha(folder/'native_driver_rng.pt') == marker['rng_sha256']
        for name, info in files.items():
            assert (folder/name).stat().st_size == info['size'], name
            assert sha(folder/name) == info['sha256'], name
        proofs[str(number)] = dict(files=len(files), all_sizes_and_sha256_verified=True,
                                   method_last_iteration=runtime['gradus_state']['last_iteration'])
    folder = folders[step]
    ranks = []
    for rank in (0, 1):
        optimizer = torch.load(folder/f'actor/optim_world_size_2_rank_{rank}.pt',
                               map_location='cpu', weights_only=False, mmap=True)
        steps = {int(s['step']) for s in optimizer['state'].values()}
        assert steps == {previous[rank][0]}
        nonzero = False
        for state in optimizer['state'].values():
            for name, tensor in state.items():
                if torch.is_tensor(tensor):
                    assert torch.isfinite(tensor).all().item(), (rank, name)
                    if name == 'exp_avg':
                        nonzero |= bool(torch.count_nonzero(tensor).item())
        assert nonzero
        extra = torch.load(folder/f'actor/extra_state_world_size_2_rank_{rank}.pt',
                           map_location='cpu', weights_only=False)
        assert extra['lr_scheduler']['last_epoch'] == step and 'rng' in extra
        ranks.append(dict(rank=rank, optimizer_step=steps.pop(), scheduler_epoch=step,
                          all_optimizer_tensors_finite=True, nonzero_first_moment=True, rng_present=True))
        del optimizer, extra
        gc.collect()
    driver = torch.load(folder/'native_driver_rng.pt', map_location='cpu', weights_only=False)
    assert {'python', 'numpy', 'torch', 'validation_dataloader', 'environment_workers'} <= driver.keys()
    workers = {k: len(v) for k, v in driver['environment_workers'].items()}
    assert workers == {'train': 128, 'validation': 128}
    assert all({'python', 'numpy', 'torch', 'user_sessions'} <= state.keys()
               for states in driver['environment_workers'].values() for state in states)
    runtime = read(folder/'native_runtime.json')
    assert set(runtime['task_cursors']) == {'train', 'validation'}
    if final:
        assert read(ROOT/'actual_resume_verification.json')['status'] == 'PASS'
    if not final:
        saved = read(ROOT/'checkpoints/best_checkpoints/global_step_116/native_runtime.json')
        assert len(runtime['gradus_state']['history']) == 4
        assert runtime['gradus_state']['history'][:-1] == saved['gradus_state']['history'][1:]
        assert read(ROOT/'gradus/diagnostics/iteration_117.json')['metrics']['history_iterations'] == 4
    return dict(at=datetime.now(timezone.utc).isoformat(), status='PASS', final=final,
                verified_step=step, config_and_release_unchanged=True, completed_iterations=completed,
                resume_decision=selected[0], actual_update_audit=audited, ranks=ranks,
                checkpoints=proofs, best=index['best'], latest=index['latest'],
                pinned_steps=index['pinned_steps'], worker_states=workers,
                driver_rng_and_cursors_and_dataloader_verified=True,
                method_history_continuation_verified=True,
                limitation='Reload and continuation verified; this does not establish bitwise identity to an uninterrupted GPU run.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    result = verify(args.final)
    destination = ROOT/('final_training_verification.json' if args.final else 'actual_resume_verification.json')
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
