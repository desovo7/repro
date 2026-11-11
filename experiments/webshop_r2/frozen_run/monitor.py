"""Read-only resource sampling for a WebShop run; exits when the given service stops. Same fields as the run's monitor.py.

usage: webshop_monitor.py UNIT OUTPUT.jsonl
"""
import datetime
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

unit, output = sys.argv[1], Path(sys.argv[2])
while True:
    state = subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '--value'], text=True).strip()
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value, *_ = line.split()
        if name in ['MemTotal:', 'MemAvailable:', 'MemFree:', 'Cached:', 'SwapFree:']:
            memory[name.rstrip(':')] = int(value) / 1024 ** 2
    groups = defaultdict(lambda: dict(processes=0, pss_gib=0.0, rss_gib=0.0))
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            if unit.removesuffix('.service') not in (p / 'cgroup').read_text():
                continue
            cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if 'WebshopWorker' in cmd:
                kind = 'webshop_env'
            elif 'ray::WorkerDict' in cmd:
                kind = 'model_workers'
            elif 'ray::Runner' in cmd:
                kind = 'trainer'
            else:
                continue
            stats = {l.split(':')[0]: int(l.split()[1]) for l in (p / 'smaps_rollup').read_text().splitlines()
                     if l.startswith(('Rss:', 'Pss:'))}
            groups[kind]['processes'] += 1
            groups[kind]['pss_gib'] += stats.get('Pss', 0) / 1024 ** 2
            groups[kind]['rss_gib'] += stats.get('Rss', 0) / 1024 ** 2
        except (OSError, ValueError):
            pass
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'],
                                  text=True).strip().splitlines()
    with output.open('a') as out:
        out.write(json.dumps(dict(at=datetime.datetime.now().astimezone().isoformat(), state=state, memory_gib=memory,
                                  process_groups=groups, gpu=gpu)) + '\n')
    if state not in ['active', 'activating']:
        break
    time.sleep(30)
