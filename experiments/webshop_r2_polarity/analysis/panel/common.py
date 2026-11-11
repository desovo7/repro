"""Frozen, evaluation-only WebShop panel utilities (no training imports)."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = Path('/data_b/qzluo/workspace/wlx/Repro-verl-agent')
R2 = BASE / 'runs/gradus-webshop-h800-s0-polarity-from115-20261111'
GIGPO = BASE / 'runs/gigpo-webshop-h800-1p5b-native-jvmcap-20261108'
SOURCE = R2 / 'source'
JVM = '-Xmx128m -Xms16m -XX:+UseSerialGC'
GPU_UUIDS = ['GPU-8fc2df35-0581-a2e9-ae6a-0a0acbcad989', 'GPU-e45494c2-3593-f188-c0c2-0f29a06ba8b3']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def canonical(value):
    def convert(x):
        if isinstance(x, set):
            return sorted(x)
        if hasattr(x, 'item'):
            return x.item()
        raise TypeError(type(x).__name__)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=convert)


def value_hash(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def request_seed(task_uid, replicate_seed, turn):
    return int(value_hash(['webshop-fixed-panel-v1', task_uid, replicate_seed, turn])[:16], 16) % (2**63-1)


def resources(require_gpus=True):
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available_gib = int(mem['MemAvailable'].split()[0]) / 1024**2
    if available_gib < 190:
        raise RuntimeError(f'Need190GiB available for128native environment workers; have{available_gib:.1f}')
    output = {'observed_at': now(), 'available_gib': available_gib, 'gpus': []}
    if require_gpus:
        for index, expected in enumerate(GPU_UUIDS):
            raw = subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
            uuid, free = [x.strip() for x in raw.split(',')]
            apps = subprocess.check_output(['nvidia-smi', '-i', str(index), '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
            if uuid != expected or int(free) < 76000 or apps:
                raise RuntimeError(f'GPU{index} changed or occupied: {uuid}, {free}, {apps}')
            output['gpus'].append({'index': index, 'uuid': uuid, 'free_mib': int(free)})
    return output


def verify_frozen():
    card = read(ROOT / 'experiment_card.json')
    for name, expected in card['frozen_files'].items():
        if digest(name) != expected:
            raise ValueError('Frozen evaluation input changed: ' + name)
    return card


def checkpoint_info(run, step, verify_models=True):
    folder = run / f'checkpoints/best_checkpoints/global_step_{step}'
    marker = read(folder / 'NATIVE_COMPLETE.json')
    manifest = read(folder / 'native_files.json')
    if marker['iteration_id'] != step or marker['world_size'] != 2:
        raise ValueError('Checkpoint identity or world size mismatch')
    if digest(folder / 'native_files.json') != marker['checkpoint_manifest_sha256']:
        raise ValueError('Manifest digest mismatch')
    if digest(folder / 'native_runtime.json') != marker['runtime_sha256']:
        raise ValueError('Runtime digest mismatch')
    for name, item in manifest.items():
        if (folder / name).stat().st_size != item['size']:
            raise ValueError('Checkpoint file size mismatch: ' + name)
    actor_files = {}
    for name, item in manifest.items():
        if name.startswith('actor/') and not any(x in name for x in ['optim_', 'extra_state_']):
            if verify_models and digest(folder / name) != item['sha256']:
                raise ValueError('Checkpoint model/tokenizer digest mismatch: ' + name)
            actor_files[name] = item
    if len([n for n in actor_files if 'model_world_size_' in n]) != 2:
        raise ValueError('Expected two FSDP model shards')
    return {'run_id': run.name, 'step': step, 'folder': str(folder), 'manifest_sha256': digest(folder / 'native_files.json'), 'actor_files': actor_files}
