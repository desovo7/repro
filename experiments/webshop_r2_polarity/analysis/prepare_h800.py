"""Verify and fork a complete aligned checkpoint into an isolated intervention run."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from omegaconf import OmegaConf

STAGING = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(STAGING))
from webshop.core import Config, SCHEMA
from webshop.entry import validate_config, verify_release
from webshop.native_retention import CheckpointRetention, REQUIRED, digest, write_json

BASE = Path('/data_b/qzluo/workspace/wlx/Repro-verl-agent')
SOURCE = BASE / 'runs/gradus-webshop-h800-s0-r2aligned-20261109'
PARENT = SOURCE / 'checkpoints/best_checkpoints/global_step_115'
TARGET = BASE / 'runs/gradus-webshop-h800-s0-polarity-from115-20261111'
MODULES = ['__init__.py', 'core.py', 'adapter.py', 'r2_credit.py', 'integration.py', 'entry.py', 'persistence.py', 'native_retention.py', 'actor_audit.py']


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


assert not TARGET.exists(), 'Refuse to overwrite a run'
original = json.loads((SOURCE / 'config.json').read_text())
source_runtime = json.loads((PARENT / 'native_runtime.json').read_text())
source_marker = json.loads((PARENT / 'NATIVE_COMPLETE.json').read_text())
source_marker_hash = sha(PARENT / 'NATIVE_COMPLETE.json')
source_manifest = json.loads((PARENT / 'native_files.json').read_text())
assert REQUIRED <= {str(p.relative_to(PARENT)) for p in PARENT.rglob('*') if p.is_file()}
assert source_runtime['iteration_id'] == source_runtime['gradus_state']['last_iteration'] == 115
assert source_runtime['gradus_state']['schema'] == 'gradus-webshop-r2-aligned-v1'
assert source_runtime['gradus_state']['config'] == original['gradus_webshop']
assert sha(PARENT / 'native_files.json') == source_marker['checkpoint_manifest_sha256']
assert sha(PARENT / 'native_runtime.json') == source_marker['runtime_sha256']
for relative, expected in source_manifest.items():
    p = PARENT / relative
    assert p.stat().st_size == expected['size'] and sha(p) == expected['sha256'], relative
print('PARENT_FULL_HASH_VERIFIED', len(source_manifest), flush=True)


def rename(v):
    if isinstance(v, dict): return {k:rename(x) for k,x in v.items()}
    if isinstance(v, list): return [rename(x) for x in v]
    if isinstance(v, str): return v.replace(str(SOURCE), str(TARGET))
    return v


cfg = rename(original)
cfg['gradus_webshop'] = asdict(Config(**original['gradus_webshop'], gate_invalid_noop=True,
                                    return_credit_mode='neutral', neutral_terminal_buy=True))
cfg['trainer']['experiment_name'] = TARGET.name
cfg['trainer']['total_training_steps'] = 150
cfg['trainer']['resume_mode'] = 'auto'
cfg['trainer']['resume_from_path'] = None
cfg['trainer']['val_before_train'] = False
cfg['native_persistence']['diagnostic_steps'] = [115, 116]
cfg['ray_init']['_temp_dir'] = '/tmp/ws-polarity-h800-20261111'
cfg['webshop_release']['files'] = {n:sha(STAGING / 'webshop' / n) for n in MODULES}
identity = cfg['native_persistence']['identity']
identity.update(run_id=TARGET.name, config_hash='pending')
identity['config_hash'] = hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
validate_config(OmegaConf.create(cfg))

TARGET.mkdir()
for folder in ['source', 'data', 'deps', 'deps-direct']:
    # Immutable native inputs stay byte-identical; no edits through these links.
    shutil.copytree(SOURCE / folder, TARGET / folder, copy_function=os.link, ignore=shutil.ignore_patterns('__pycache__'))
(TARGET / 'webshop').mkdir()
for n in MODULES: shutil.copy2(STAGING / 'webshop' / n, TARGET / 'webshop' / n)
write_json(TARGET / 'config.json', cfg)
environment = (SOURCE / 'environment.sh').read_text().replace(str(SOURCE), str(TARGET))
environment = environment.replace('/tmp/ws-gradus-h800-r2aligned-20261109', '/tmp/ws-polarity-h800-20261111')
environment += '\nexport PYTHONDONTWRITEBYTECODE=1\n'
(TARGET / 'environment.sh').write_text(environment)
(TARGET / 'launch.sh').write_text((SOURCE / 'launch.sh').read_text().replace('RuntimeMaxSec=24h', 'RuntimeMaxSec=8h'))
shutil.copy2(SOURCE / 'monitor.py', TARGET / 'monitor.py')

dest = TARGET / 'checkpoints/global_step_115'
dest.mkdir(parents=True)
for path in PARENT.rglob('*'):
    if not path.is_file() or path.name in {'native_runtime.json','NATIVE_COMPLETE.json','native_files.json','RETENTION_COMPLETE.json'}:
        continue
    q = dest / path.relative_to(PARENT)
    q.parent.mkdir(parents=True, exist_ok=True)
    os.link(path, q)
runtime = json.loads(json.dumps(source_runtime))
runtime.update(identity)
runtime['method'] = 'WebShop R2 auxiliary eligibility with neutral return and terminal buy'
runtime['gradus_state']['schema'] = SCHEMA
runtime['gradus_state']['config'] = cfg['gradus_webshop']
assert runtime['gradus_state']['history'] == source_runtime['gradus_state']['history']
write_json(dest / 'native_runtime.json', runtime)
manifest = {}
for p in sorted(dest.rglob('*')):
    if not p.is_file(): continue
    rel = str(p.relative_to(dest))
    if rel == 'native_runtime.json':
        manifest[rel] = dict(size=p.stat().st_size, sha256=sha(p))
    else:
        assert (p.stat().st_dev, p.stat().st_ino) == ((PARENT/rel).stat().st_dev, (PARENT/rel).stat().st_ino)
        manifest[rel] = source_manifest[rel]
write_json(dest / 'native_files.json', manifest)
write_json(dest / 'NATIVE_COMPLETE.json', dict(runtime,
    runtime_sha256=sha(dest/'native_runtime.json'), rng_sha256=sha(dest/'native_driver_rng.pt'),
    checkpoint_manifest_sha256=sha(dest/'native_files.json')))
(TARGET/'checkpoints/latest_checkpointed_iteration.txt').write_text('115')
retention = CheckpointRetention(TARGET/'checkpoints', identity, [115,116])
retention._check_complete(dest,115)
retention._check_archive(115)

source_after = {n:sha(PARENT/n) for n in ['NATIVE_COMPLETE.json','native_runtime.json','native_files.json']}
assert source_after['NATIVE_COMPLETE.json'] == source_marker_hash
assert source_after['native_runtime.json'] == source_marker['runtime_sha256']
assert source_after['native_files.json'] == source_marker['checkpoint_manifest_sha256']
write_json(TARGET/'parent_checkpoint_migration.json', dict(at=datetime.now(timezone.utc).isoformat(),
    status='PASS', parent=str(PARENT), parent_run_id=source_runtime['run_id'], parent_iteration=115,
    parent_full_manifest=source_manifest, parent_marker_sha256=source_after['NATIVE_COMPLETE.json'],
    parent_code_hashes=original['webshop_release']['files'], inherited_files_byte_identical=True,
    inherited_model_optimizer_rank_rng_driver_rng_data_cursors=True, history_identical=True,
    changed_fields=['run_id','config_hash','method label','gradus_state.schema','gradus_state.config'],
    method_contract_migration='Explicit intervention fork; not an unchanged-method continuation',
    new_identity=identity, source_checkpoint_unmodified=True))
write_json(TARGET/'experiment_card.json', dict(at=datetime.now(timezone.utc).isoformat(),
    status='FROZEN_PRE_GPU', run_id=TARGET.name, parent_checkpoint=str(PARENT),
    authorization='User selected neutral return auxiliary credit and aligned complete step115 continuation to150.',
    start_step=115, final_step=150, additional_training_iterations=35,
    method=cfg['gradus_webshop'], primary='Fixed final150 on existing512tasks x3 repeats, success rate',
    targets=dict(format_invalid_count_lt=50, format_invalid_rate_lt=.006, return_navigation_all_train_gt=0,
                 option_loop_count=0, timeout_rate_le=.04, success_count_gt=1085),
    baseline='Frozen aligned150 and local GiGPO150; single training-seed development comparison',
    gpu_indices=[0,1], training_wall_hours_estimate=[3,4.5], training_wall_hours_limit=8,
    training_gpu_hours_estimate=[6,9], estimate_basis='Prior150round training ~12–14h on same2GPU host;35round portion plus full restore checks.',
    panel_additional_episodes=1536, panel_wall_hours_limit=3, monetary_cost_unknown=True,
    checkpoint_retention=dict(best=3,metric='val/success_rate',tie_break='later iteration first',latest_full=True,pinned=[115,116]),
    acceptance='Final CPU replay and CPU tests pass before start; actual116 save,stop,reload117 verification before full continuation.',
    limits='Three-intervention combination; no single-component causal claim, no unseen-panel or multi-training-seed claim.'))
for folder, manifest_key in [('webshop','files'),('source','source'),('data','data')]:
    for relative, expected in cfg['webshop_release'][manifest_key].items():
        assert sha(TARGET/folder/relative) == expected, (folder, relative)
write_json(TARGET/'PREPARED.json', dict(status='prepared_not_started', at=datetime.now(timezone.utc).isoformat(),
    parent_verified=True, source_checkpoint_unmodified=True, run=str(TARGET), config_hash=identity['config_hash'],
    method_files=cfg['webshop_release']['files']))
print(json.dumps(dict(status='PREPARED',run=str(TARGET),parent=str(PARENT),config_hash=identity['config_hash'])),flush=True)
