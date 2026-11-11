"""Read-only integrity and optimizer update audit, executed on H800 after step 1."""
from pathlib import Path
import datetime
import gc
import hashlib
import json
import os
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

torch.set_num_threads(1)
ROOT = Path('/data_b/qzluo/workspace/wlx/Repro-verl-agent/runs/gradus-webshop-h800-s0-polarity-from115-20261111')
FOLDER = ROOT / 'checkpoints/global_step_116'
assert (FOLDER / 'NATIVE_COMPLETE.json').exists(), 'First complete checkpoint not ready'
manifest = json.loads((FOLDER / 'native_files.json').read_text())
files = {}
for relative, expected in manifest.items():
    path = FOLDER / relative
    assert path.stat().st_size == expected['size'], relative
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert actual == expected['sha256'], relative
    if path.suffix == '.pt':
        with zipfile.ZipFile(path) as archive:
            assert any(name.endswith('/data.pkl') for name in archive.namelist()), relative
    files[relative] = expected

parent = ROOT / 'checkpoints/best_checkpoints/global_step_115'
parent_steps = []
for rank in (0, 1):
    opt = torch.load(parent / 'actor' / f'optim_world_size_2_rank_{rank}.pt', map_location='cpu', weights_only=False, mmap=True)
    parent_steps.append(sorted({int(v['step']) for v in opt['state'].values()}))
    del opt
assert parent_steps[0] == parent_steps[1] and len(parent_steps[0]) == 1
ranks = []
for rank in (0, 1):
    optimizer = torch.load(FOLDER / 'actor' / f'optim_world_size_2_rank_{rank}.pt',
                           map_location='cpu', weights_only=False, mmap=True)
    steps, moment_nonzero, tensors = set(), False, 0
    for state in optimizer['state'].values():
        steps.add(int(state['step']))
        for key, tensor in state.items():
            if torch.is_tensor(tensor):
                assert torch.isfinite(tensor).all().item(), (rank, key)
                tensors += 1
                if key == 'exp_avg':
                    moment_nonzero |= bool(torch.count_nonzero(tensor).item())
    assert len(steps) == 1 and min(steps) > 0 and moment_nonzero, (rank, steps)
    extra = torch.load(FOLDER / 'actor' / f'extra_state_world_size_2_rank_{rank}.pt',
                      map_location='cpu', weights_only=False)
    assert 'rng' in extra and 'lr_scheduler' in extra
    ranks.append(dict(rank=rank, optimizer_steps=sorted(steps), all_optimizer_tensors_finite=True,
                      optimizer_tensors=tensors, nonzero_first_moment=moment_nonzero,
                      scheduler_epoch=extra['lr_scheduler']['last_epoch'], rng_present=True))
    del optimizer, extra
    gc.collect()
assert ranks[0]['optimizer_steps'] == ranks[1]['optimizer_steps']
assert ranks[0]['optimizer_steps'][0] > parent_steps[0][0]
driver = torch.load(FOLDER / 'native_driver_rng.pt', map_location='cpu', weights_only=False)
runtime = json.loads((FOLDER / 'native_runtime.json').read_text())
assert runtime['iteration_id'] == 116 and runtime['world_size'] == 2
assert runtime['gradus_state']['last_iteration'] == 116
diag = json.loads((ROOT / 'gradus/diagnostics/iteration_116.json').read_text())
assert diag['audit']['all_ranks_finite'] and diag['audit']['world_size'] == 2
assert diag['audit']['actual_steps_min'] == diag['audit']['expected_steps_min'] > 0
assert ranks[0]['optimizer_steps'][0] - parent_steps[0][0] == diag['audit']['actual_steps_min']
assert diag['metrics']['auxiliary_gated_nonzero_tokens'] == 0
assert set(runtime['task_cursors']) == {'train', 'validation'}
assert {'python', 'numpy', 'torch', 'environment_workers', 'validation_dataloader'} <= driver.keys()
worker_counts = {k:len(v) for k,v in driver['environment_workers'].items()}
assert worker_counts == {'train':128, 'validation':128}, worker_counts
assert all({'python','numpy','torch','user_sessions'} <= state.keys()
           for rows in driver['environment_workers'].values() for state in rows)
retention = json.loads((ROOT / 'checkpoints/best_checkpoints/index.json').read_text())
archive = ROOT / 'checkpoints/best_checkpoints/global_step_116'
assert retention['latest']['step'] >= 116
assert retention['pinned_steps'] == [115,116]
assert runtime['gradus_state']['schema'] == 'gradus-webshop-r2-polarity-v1'
links = {}
if archive.exists():
    assert (archive / 'RETENTION_COMPLETE.json').exists()
    for name in manifest:
        a, b = (FOLDER / name).stat(), (archive / name).stat()
        links[name] = (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)
    assert all(links.values())
report = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), step=116,
              checked_files=files, ranks=ranks, parent_optimizer_steps=parent_steps, worker_counts=worker_counts,
              task_cursors_saved=True, driver_rng_saved=True, dataloader_saved=True,
              retention=retention, protected_hard_links=links,
              gradus_state_verified=True, method_diagnostics=diag, gpu_resume_test_performed=False,
              limitation='File integrity and saved update state verified; actual GPU reload is the next integration check.')
(ROOT / 'first_checkpoint_verified.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['checked_files','protected_hard_links']}, ensure_ascii=False))
