#!/usr/bin/env python3
"""Recount all published per-episode metrics; no model inference or raw-text claims."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/gradus/evidence/polarity_20261111'
data = json.loads((BASE/'panel/metrics.json').read_text())
result = json.loads((BASE/'results.json').read_text())
protocol = json.loads((BASE/'panel/protocol.json').read_text())
assert len(data['task_uids']) == len(set(data['task_uids'])) == 512
assert data['replicate_seeds'] == [0, 1, 2]
columns = data['columns']
arrays = {}
for tag, model in data['models'].items():
    rows = np.asarray(model['rows'], dtype=np.float64)
    assert rows.shape == (1536, len(columns)) and np.isfinite(rows).all()
    assert len(model['raw_batch_sha256']) == 12
    assert model['task_manifest_sha256'] == protocol['task_manifest_sha256']
    assert np.array_equal(rows[:, columns.index('success')], rows[:, columns.index('score')] == 1)
    assert np.all((rows[:, columns.index('score')] >= 0) & (rows[:, columns.index('score')] <= 1))
    for key in ['success','timeout','failed_buy','option_loop']:
        assert set(rows[:, columns.index(key)]) <= {0, 1}
    expected = result['panel'][tag]
    for key, value in expected['counts'].items():
        assert rows[:, columns.index(key)].sum() == value, (tag, key)
    for key, value in expected['means'].items():
        assert abs(rows[:, columns.index(key)].mean()-value) < 1e-12, (tag, key)
    for seed, expected_means in expected['by_seed'].items():
        for key, value in expected_means.items():
            assert abs(rows[int(seed)::3, columns.index(key)].mean()-value) < 1e-12
    arrays[tag] = rows.reshape(512, 3, len(columns)).mean(axis=1)

indices = np.random.default_rng(20261109).integers(0, 512, size=(10000, 512))
assert len(result['paired_contrasts']) == 2
for contrast in result['paired_contrasts']:
    delta = arrays[contrast['candidate']] - arrays[contrast['reference']]
    for key, expected in contrast['metrics'].items():
        values = delta[:, columns.index(key)]
        assert abs(values.mean()-expected['difference']) < 1e-12
        ci = np.quantile(values[indices].mean(axis=1), [.025, .975])
        assert np.allclose(ci, expected['ci95'], atol=1e-12, rtol=0)

c = result['panel']['r2_150']['counts']
observed = {'format_invalid_count': c['format_invalid_actions'],
            'format_invalid_rate': c['format_invalid_actions']/c['steps'],
            'new116_150_return_navigation': result['return_navigation']['intervention116_150'],
            'option_loop_episodes': c['option_loop'], 'timeout_rate': c['timeout']/1536,
            'successes': c['success']}
predicates = {'format_invalid_count': lambda x: x < 50, 'format_invalid_rate': lambda x: x < .006,
              'new116_150_return_navigation': lambda x: x > 0, 'option_loop_episodes': lambda x: x == 0,
              'timeout_rate': lambda x: x <= .04, 'successes': lambda x: x > 1085}
for target in result['targets']:
    assert target['observed'] == observed[target['name']]
    assert target['passed'] == predicates[target['name']](observed[target['name']])
assert result['all_exploratory_targets_met'] == all(t['passed'] for t in result['targets'])
assert result['new_training_iterations'] == 35 and result['parent'] == 'aligned complete115'
print('PASS: 4608 metric rows; counts, means, three repeats, two paired contrasts, and target decisions')
print('Scope: published metric recount; full raw-observation extraction was verified before publication.')
