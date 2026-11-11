"""CPU-only paired replay of the frozen corpus with independent switches."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time


def package(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path) / '__init__.py', submodule_search_locations=[str(path)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return importlib.import_module(name + '.core')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--final-check', action='store_true')
    args = ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    base, out = Path(args.base), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    run_ids = {'old': 'gradus-webshop-h800-s0-r2-jvmcap-20261108', 'aligned': 'gradus-webshop-h800-s0-r2aligned-20261109'}
    baseline = package('replay_baseline', base / 'runs' / run_ids['aligned'] / 'webshop')
    candidate = package('replay_candidate', args.candidate)
    frozen_config = json.loads((base / 'runs' / run_ids['aligned'] / 'config.json').read_text())['gradus_webshop']
    configs = {
        'disabled': candidate.Config(**frozen_config),
        'eligibility_only': candidate.Config(**frozen_config, gate_invalid_noop=True),
        'return_only': candidate.Config(**frozen_config, return_credit_mode='neutral'),
        'buy_only': candidate.Config(**frozen_config, neutral_terminal_buy=True),
        'combined': candidate.Config(**frozen_config, gate_invalid_noop=True, return_credit_mode='neutral', neutral_terminal_buy=True),
    }
    if args.final_check:
        configs = {k:v for k,v in configs.items() if k in {'disabled','combined'}}
    summaries = {}
    for label, run_id in run_ids.items():
        start = time.monotonic()
        root = base / 'runs' / run_id
        native = baseline.CreditEngine(baseline.Config(**frozen_config))
        engines = {k: candidate.CreditEngine(c) for k, c in configs.items()}
        stats = defaultdict(lambda: Counter(n=0, positive=0, robust_positive=0, negative=0, nonzero=0, sum_centered=0., sum_raw=0.))
        counts, all_navigation = Counter(), Counter()
        rounds, examples = [], {}
        h = hashlib.sha256()
        def rows():
            with (root / 'gradus/train_traces.jsonl').open('rb') as f:
                for line in f:
                    h.update(line)
                    yield json.loads(line)
        for iteration, group in itertools.groupby(rows(), key=lambda r: r['iteration_id']):
            batch = list(group)
            assert iteration == len(rounds) + 1 and len(batch) == 128
            original = native.prepare(batch, iteration)
            variants = {k: e.prepare(batch, iteration) for k, e in engines.items()}
            disabled = variants['disabled']
            for field in ['raw', 'deltas', 'centered', 'af', 'diagnostics', 'action_groups']:
                assert disabled[field] == original[field], (label, iteration, 'disabled mismatch', field)
            for key, engine in engines.items():
                assert engine.table_snapshot == native.table_snapshot, (label, iteration, key, 'tables changed')
                assert engine.pending[1] == native.pending[1], (label, iteration, key, 'history changed')
            if label == 'aligned':
                saved = json.loads((root / 'gradus/diagnostics' / f'iteration_{iteration}.json').read_text())['metrics']
                for key, val in original['diagnostics'].items():
                    assert math.isclose(val, saved[key], rel_tol=1e-10, abs_tol=1e-10), (iteration, key)
            cohort_counts = Counter()
            for raw in batch:
                t = candidate.parse_trace(raw, configs['disabled'])['trace']
                counts['trajectories'] += 1
                for i, step in enumerate(raw['steps']):
                    uid = step['action_uid']
                    counts['all_actions'] += 1
                    phase = '100_135' if 100 <= iteration <= 135 else '136_150' if 136 <= iteration <= 150 else 'other'
                    if t.return_navs[i]:
                        all_navigation['all150'] += 1
                        all_navigation[phase] += 1
                        if 101 <= iteration <= 150:
                            all_navigation['101_150'] += 1
                    if not original['af'][t.group_uid]:
                        assert all(r['centered'][uid] == 0 for r in variants.values())
                        continue
                    cohorts = ['all_AF']
                    if t.noops[i]: cohorts.append('noop')
                    if not t.format_valids[i]: cohorts.append('format_invalid')
                    if not t.format_valids[i] and not t.noops[i]: cohorts.append('format_invalid_not_noop')
                    if t.noops[i] and t.format_valids[i]: cohorts.append('noop_not_format_invalid')
                    if t.return_navs[i]: cohorts.append('return_navigation')
                    if t.terminal_buys[i]: cohorts.append('terminal_buy')
                    for cohort in cohorts:
                        cohort_counts[cohort] += 1
                        for variant, result in [('baseline', original)] + list(variants.items()):
                            value, raw_value = result['centered'][uid], result['raw'][uid]
                            assert math.isfinite(value) and math.isfinite(raw_value)
                            s = stats[(variant, cohort)]
                            s.update(n=1, positive=int(value > 0), robust_positive=int(value > 1e-12),
                                     negative=int(value < 0), nonzero=int(value != 0), sum_centered=value, sum_raw=raw_value)
                    combined = variants['combined']
                    if t.noops[i] or not t.format_valids[i] or t.return_navs[i] or t.terminal_buys[i]:
                        assert all(combined[f][uid] == 0 for f in ['deltas', 'raw', 'centered']), (label, uid, 'gate failed')
                    if 'eligibility_only' in variants and (t.noops[i] or not t.format_valids[i]):
                        assert variants['eligibility_only']['centered'][uid] == 0
                    if 'return_only' in variants and t.return_navs[i]:
                        assert variants['return_only']['raw'][uid] == variants['return_only']['centered'][uid] == 0
                    if 'buy_only' in variants and t.terminal_buys[i]:
                        assert variants['buy_only']['centered'][uid] == 0
                    if label == 'aligned' and iteration == 8 and t.group_uid == 'e5de3c22-4e69-40e7-9185-38d12e4e33db' and t.states[i] == '["item","within",["unresolved","selected_requested"]]':
                        examples.setdefault('sample_C_bucket', []).append(dict(uid=uid, action=step['action'], noop=t.noops[i], format_valid=t.format_valids[i], return_nav=t.return_navs[i], terminal_buy=t.terminal_buys[i],
                            baseline={k:original[k][uid] for k in ['deltas','raw','centered']},
                            variants={name:{k:result[k][uid] for k in ['deltas','raw','centered']} for name,result in variants.items() if name != 'disabled'}))
            # Exercise the exact action->token function with heterogeneous weights;
            # this validates zeros, but is NOT an estimate of actual new token scale.
            uids = list(variants['combined']['centered'])
            injections, metrics = candidate.token_injections(variants['combined'], uids, [1 + i % 7 for i in range(len(uids))], configs['combined'])
            for uid, v in zip(uids, injections):
                assert math.isfinite(v) and abs(v) <= 5
                if uid in variants['combined']['gate_reasons']:
                    assert v == 0
            for name, r in variants.items():
                assert abs(math.fsum(r['centered'].values())) < 1e-8
            rounds.append(dict(iteration=iteration, cohorts=dict(cohort_counts), diagnostics={k:r['diagnostics'] for k,r in variants.items()},
                               combined_changed_actions=sum(original['centered'][u] != variants['combined']['centered'][u] for u in uids)))
            native.commit(iteration)
            for engine in engines.values(): engine.commit(iteration)
            if iteration % 25 == 0:
                print(label, iteration, 'iterations verified', round(time.monotonic()-start,1), 'seconds', flush=True)
        assert len(rounds) == 150 and counts['trajectories'] == 19200
        report_stats = {}
        for (variant, cohort), v in stats.items():
            report_stats.setdefault(variant, {})[cohort] = dict(v, mean_centered=v['sum_centered']/v['n'], mean_raw=v['sum_raw']/v['n'])
        assert report_stats['combined']['all_AF']['nonzero'] > 0, 'The whole method was accidentally disabled'
        if label == 'aligned':
            assert report_stats['baseline']['format_invalid_not_noop']['n'] == 10247
            assert report_stats['baseline']['format_invalid_not_noop']['positive'] == 4019
            assert len(examples['sample_C_bucket']) == 15
            assert all(x['variants']['combined']['raw'] == x['variants']['combined']['centered'] == 0 for x in examples['sample_C_bucket'])
        report = dict(status='PASS', corpus=label, run_id=run_id, iterations=150, counts=dict(counts),
                      elapsed_cpu_wall_seconds=time.monotonic()-start, train_traces_sha256=h.hexdigest(),
                      configs={k:asdict(v) for k,v in configs.items()}, baseline_disabled_exact_match=True,
                      history_tables_unchanged=True, non_AF_zero=True, centered_bucket_sums_zero=True,
                      token_gate_zero_with_synthetic_weights=True, actual_new_token_scale_measured=False,
                      cohorts=report_stats, return_navigation_all_trajectories=dict(all_navigation), examples=examples, rounds=rounds,
                      scope='Aligned core paired replay on frozen trajectories; old corpus is counterfactual and not the deployed old algorithm. No policy update or causal performance evidence.')
        (out/f'{label}_replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        summaries[label] = {k:report[k] for k in ['status','iterations','counts','elapsed_cpu_wall_seconds','train_traces_sha256','cohorts','return_navigation_all_trajectories']}
        print(label, 'PASS', flush=True)
    (out/'SUMMARY.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    main()
